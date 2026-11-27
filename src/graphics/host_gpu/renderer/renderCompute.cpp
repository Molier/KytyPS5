#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/frameCost.h"
#include "common/ports.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/ports.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/dispatchGuard.h"
#include "graphics/host_gpu/renderer/dispatchLimits.h"
#include "graphics/host_gpu/renderer/dispatchOrigin.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/bufferProbe.h"
#include "graphics/host_gpu/renderer/pipeline/frameHoles.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/pipeline/skippedDraws.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const auto& program   = input.stage.module->info;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	// Offset-or-swizzle bounds do not describe the element range proved by the shader.
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.OutOfBounds() == 3 || descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	const auto elements = descriptor.NumRecords();
	const bool covered = fill.bounded_elements != 0
	                         ? fill.bounded_elements == elements && invocations >= elements
	                         : invocations == elements;
	if (!covered || invocations > uint64_t {UINT32_MAX} + 1 || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeFill(const ShaderComputeInputInfo& input,
                                          CommandBuffer& command, uint32_t group_x,
                                          uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = input.stage.module->info;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	const bool image_clear =
	    cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear);
	if (!image_clear) {
		// CPU fills invalidate overlapping cached images. Preserve the native dispatch for
		// partial or ambiguous image backings, including pooled metadata aliases.
		if ((descriptor.Base48() & 3u) != 0 ||
		    !m_context.IsMapped(descriptor.Base48(), size) ||
		    cache.IsRegionRegistered(descriptor.Base48(), size)) {
			return false;
		}
		// Metadata need not have an image backing yet. Keep proven fills in guest memory
		// when CPU-owned, so color-clear discovery can read them without a GPU download.
		m_context.GetBufferCache().FillBuffer(descriptor.Base48(), size, packed_clear, false);
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute %s shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     image_clear ? "image clear" : "buffer fill", input.stage.module->info.shader_hash,
		     descriptor.Base48(), size, packed_clear);
	}
	return true;
}

// Global loads read pages without a buffer as zero. Cache the 32 KiB after each base the shader
// takes from user data or scalar loads before the dispatch reads it.
static void CacheDmaBases(RenderContext& context, const ShaderStageRuntime& stage) {
	const auto back = [&context](uint32_t low, uint32_t high) {
		const auto base = low | uint64_t {high} << 32u;
		for (uint64_t index = 0; index <= 2; ++index) {
			const auto page = base + index * BufferCache::CACHING_PAGESIZE;
			if (context.IsMapped(page, 1)) (void)context.GetBufferCache().FindBuffer(page, 1);
		}
	};
	const auto& info = stage.module->info.info;
	const auto& flat = stage.resources->flattened_srt;
	const auto& user = stage.resources->user_data;
	for (const auto [low, high]: info.address_bases) back(flat[low], flat[high]);
	for (const auto [low, high]: info.user_address_bases) back(user[low], user[high]);
}

// Recognized traversals trace the mirror of the TLAS whose header address the shader loads from
// scalar data.
static void PrepareRayQuery(RenderContext& context, const ShaderStageRuntime& stage,
                            PreparedBindings& bindings) {
	const auto& program = stage.module->info;
	const auto  dword   = program.bindings.ray_query_dword;
	if (dword == ShaderRecompiler::IR::PushData::NoStart) return;
	const auto& flat       = stage.resources->flattened_srt;
	const auto [low, high] = program.info.ray_query_tlas;
	const auto address = context.GetRayTracing()->Prepare(flat[low] | uint64_t {flat[high]} << 32u);
	bindings.shader_data[dword]     = static_cast<uint32_t>(address);
	bindings.shader_data[dword + 1] = static_cast<uint32_t>(address >> 32u);
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0,
                             const vk::DispatchIndirectCommand* host_args = nullptr) {
	if (input.stage.module->info.bindings.descriptor_counts[
	        static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory)] == 0) {
		return;
	}
	auto& cache = context.GetBufferCache();
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	if (indirect_args != 0) {
		if (Config::PortEnabled(Config::Port::GpuIndirectThreadDim)) {
			if (host_args != nullptr) {
				input.workgroup_counts[0] = host_args->x;
				input.workgroup_counts[1] = host_args->y;
				input.workgroup_counts[2] = host_args->z;
			} else {
				// lds_variable indexes offset 0 by WorkgroupId and never queries OpArrayLength:
				// bind the full utility range instead of stalling the GPU for group counts.
				bindings.shared_memory = {storage.Handle(), 0, limit};
				return;
			}
		} else {
			cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand), false,
			                 Common::FrameCost::ReadbackSite::Lds);
			std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
			            sizeof(input.workgroup_counts));
		}
	}
	uint64_t size = sizeof(uint32_t);
	if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

static void CaptureComputeBufferBreadcrumbs(RenderContext&               context,
                                          const HW::ComputeShaderInfo& cs_regs,
                                          const PreparedBindings&      bindings,
                                          OpBreadcrumb&                op) {
	const auto& program = bindings.runtime->module->info;
	using BindingKind   = ShaderRecompiler::IR::DescriptorBindingKind;

	op.push_dword_count = static_cast<uint8_t>(
	    std::min<size_t>(bindings.shader_data.size(), op.push_dwords.size()));
	std::copy_n(bindings.shader_data.data(), op.push_dword_count, op.push_dwords.data());

	const size_t user_count = std::min<size_t>(
	    std::max<size_t>(cs_regs.cs_regs.user_sgpr, 16u), op.user_data_dwords.size());
	op.user_data_count = static_cast<uint8_t>(user_count);
	std::copy_n(cs_regs.cs_user_sgpr.value, user_count, op.user_data_dwords.data());

	const auto push_buf = [&](uint16_t binding, uint16_t array_element, uint64_t guest_va,
	                          uint64_t gpu_va, const vk::DescriptorBufferInfo& info,
	                          uint64_t buffer_size, bool is_storage_array,
	                          const uint8_t* mapped, uint64_t mapped_size) {
		if (op.total_buffers < UINT16_MAX) {
			++op.total_buffers;
		}
		if (op.buffer_count >= OpBreadcrumb::kMaxBuffers) {
			return;
		}
		auto& entry = op.buffers[op.buffer_count++];
		entry       = {
		          .guest_va         = guest_va,
		          .vk_buffer        = static_cast<uint64_t>(
                      reinterpret_cast<uintptr_t>(static_cast<vk::Buffer::CType>(info.buffer))),
		          .gpu_va           = gpu_va,
		          .offset           = info.offset,
		          .range            = info.range,
		          .buffer_size      = buffer_size,
		          .binding          = binding,
		          .array_element    = array_element,
		          .set              = 0,
		          .is_storage_array = static_cast<uint8_t>(is_storage_array ? 1 : 0),
		          .word_count       = 0,
		};
		if (info.range > 0 && info.range != VK_WHOLE_SIZE && info.range <= 64u * 1024u) {
			const uint64_t words = std::min<uint64_t>(info.range / sizeof(uint32_t),
			                                          entry.words.size());
			const uint64_t bytes = words * sizeof(uint32_t);
			if (bytes > 0) {
				if (guest_va != 0 &&
				    LibKernel::Memory::TryReadBacking(guest_va, entry.words.data(), bytes)) {
					entry.word_count = static_cast<uint8_t>(words);
				} else if (mapped != nullptr && info.offset <= mapped_size &&
				           bytes <= mapped_size - info.offset) {
					std::memcpy(entry.words.data(), mapped + info.offset, bytes);
					entry.word_count = static_cast<uint8_t>(words);
				}
			}
		}
	};

	const auto buffers_binding =
	    static_cast<uint16_t>(ShaderRecompiler::IR::NativeBinding(program.stage, BindingKind::Buffers));
	for (uint32_t i = 0; i < bindings.buffers.size(); ++i) {
		const bool     has_src      = i < bindings.buffer_sources.size();
		const uint64_t guest_va     = has_src ? bindings.buffer_sources[i].address : 0;
		const uint64_t backing_size = has_src ? bindings.buffer_sources[i].backing_size : 0;
		const uint64_t gpu_va       = has_src ? bindings.buffer_sources[i].gpu_va : 0;
		const uint8_t* mapped       = has_src ? bindings.buffer_sources[i].mapped : nullptr;
		const uint64_t mapped_size  = has_src ? bindings.buffer_sources[i].mapped_size : 0;
		push_buf(buffers_binding, static_cast<uint16_t>(i), guest_va, gpu_va, bindings.buffers[i],
		         backing_size, true, mapped, mapped_size);
	}

	auto&       cache      = context.GetBufferCache();
	const auto& stream_buf = cache.GetUtilityBuffer(MemoryUsage::Stream);
	const auto& local_buf  = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto  stream_map = stream_buf.Mapped();
	for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
		const auto kind = static_cast<BindingKind>(std::countr_zero(mask));
		if (kind == BindingKind::Buffers ||
		    ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
		        ShaderRecompiler::IR::ImageResourceClass::None ||
		    kind == BindingKind::Samplers) {
			continue;
		}
		const auto binding =
		    static_cast<uint16_t>(ShaderRecompiler::IR::NativeBinding(program.stage, kind));
		switch (kind) {
			case BindingKind::SharedMemory:
				push_buf(binding, 0, 0, static_cast<uint64_t>(local_buf.TryBufferDeviceAddress()),
				         bindings.shared_memory, local_buf.Size(), false, nullptr, 0);
				break;
			case BindingKind::FlattenedSrt:
				push_buf(binding, 0, 0, static_cast<uint64_t>(stream_buf.TryBufferDeviceAddress()),
				         bindings.flattened_srt, stream_buf.Size(), false,
				         stream_map.empty() ? nullptr : stream_map.data(), stream_map.size());
				break;
			case BindingKind::ShaderData:
				push_buf(binding, 0, 0, static_cast<uint64_t>(stream_buf.TryBufferDeviceAddress()),
				         bindings.shader_data_buffer, stream_buf.Size(), false,
				         stream_map.empty() ? nullptr : stream_map.data(), stream_map.size());
				break;
			case BindingKind::Gds: {
				const auto* buf = cache.GetGdsBuffer();
				push_buf(binding, 0, 0, static_cast<uint64_t>(buf->TryBufferDeviceAddress()),
				         bindings.gds, buf->Size(), false, nullptr, 0);
				break;
			}
			case BindingKind::BdaPagetable: {
				const auto* buf = cache.GetBdaPageTableBuffer();
				const vk::DescriptorBufferInfo info {buf->Handle(), 0, buf->Size()};
				push_buf(binding, 0, 0, static_cast<uint64_t>(buf->TryBufferDeviceAddress()),
				         info, buf->Size(), false, nullptr, 0);
				break;
			}
			case BindingKind::FaultBuffer: {
				const auto* buf = cache.GetFaultBuffer();
				const vk::DescriptorBufferInfo info {buf->Handle(), 0, buf->Size()};
				push_buf(binding, 0, 0, static_cast<uint64_t>(buf->TryBufferDeviceAddress()),
				         info, buf->Size(), false, nullptr, 0);
				break;
			}
			default: break;
		}
	}
}

// --buffer-probe: for dispatches of the listed shaders, compares the guest memory of every bound
// storage buffer with the bytes the GPU reads.
//
// Ordering: the probe runs on the GPU thread after the dispatch's bindings are final (the cache has
// uploaded every CPU-dirty range it was going to). It records, in the stream's current command
// buffer, one barrier, a copy of each bound range into a host-visible buffer and a second
// barrier, then submits the stream and waits for it before the dispatch itself is recorded. The
// barrier makes every earlier command of the stream (cache uploads, earlier dispatches and draws)
// visible to the copy, so the copy shows what the dispatch will start with; guest bytes are read
// right before the submit. The findings are logged before the dispatch exists, so they survive a
// hang of that dispatch. Slow by design (one stall per probed dispatch).
// The cache state is sampled BEFORE the binding step, so "synced" on a differing range means the
// cache believed the range was up to date.
namespace {

struct BufferProbeBinding {
	uint32_t                  index         = 0;
	uint64_t                  guest         = 0;
	uint64_t                  size          = 0;
	uint64_t                  descriptor_size = 0; // What the guest's descriptor declares.
	uint64_t                  compared      = 0;
	uint64_t                  staging_offset = 0;
	bool                      written       = false;
	bool                      guest_ok      = false;
	BufferProbe::CacheState   state         = BufferProbe::CacheState::Synced;
	std::vector<uint8_t>      guest_bytes;
};

class BufferProbeRunner {
public:
	static constexpr uint64_t MaxBytesPerBuffer   = 16ull * 1024 * 1024;
	static constexpr uint64_t MaxBytesPerDispatch = 64ull * 1024 * 1024;

	// Before the cache binds the buffers: remember what it believes about every bound range.
	static std::vector<BufferProbe::CacheState> SampleCacheState(RenderContext&          context,
	                                                             const PreparedBindings& bindings) {
		auto&                                     cache = context.GetBufferCache();
		std::vector<BufferProbe::CacheState> result(bindings.buffer_sources.size(),
		                                            BufferProbe::CacheState::Synced);
		for (size_t i = 0; i < bindings.buffer_sources.size(); i++) {
			const auto& source = bindings.buffer_sources[i];
			if (source.address == 0 || source.size == 0 ||
			    !GuestRange {source.address, source.size}.Valid()) {
				continue;
			}
			result[i] = BufferProbe::ClassifyCacheState(
			    cache.IsRegionCpuModified(source.address, source.size),
			    cache.IsRegionGpuModified(source.address, source.size));
		}
		return result;
	}

	void Record(RenderContext& context, CommandBuffer& command, uint64_t hash,
	            const PreparedBindings&                     bindings,
	            const std::vector<BufferProbe::CacheState>& states) {
		const auto& program = bindings.runtime->module->info;
		auto&       cache   = context.GetBufferCache();
		const auto  stream  = cache.GetUtilityBuffer(MemoryUsage::Stream).Handle();

		std::vector<BufferProbeBinding> probed;
		std::vector<vk::BufferCopy>     copies;
		std::vector<vk::Buffer>         sources;
		std::vector<uint64_t>           host_start(bindings.buffer_sources.size(), 0);
		std::vector<bool>               host_stream(bindings.buffer_sources.size(), false);
		uint64_t                        total = 0;
		const auto& snapshot = *bindings.runtime->resources;
		for (size_t resource_index = 0; resource_index < program.info.buffers.size();
		     resource_index++) {
			const auto& resource = program.info.buffers[resource_index];
			const auto  index    = resource.descriptor_index;
			if (index == UINT32_MAX || index >= bindings.buffer_sources.size() ||
			    index >= bindings.buffers.size()) {
				continue;
			}
			const auto& source = bindings.buffer_sources[index];
			const auto& info   = bindings.buffers[index];
			if (source.address == 0 || source.size == 0 || !info.buffer) continue;
			// The memory-offset adjustment the binding step packed next to the shader data.
			const auto& layout    = program.bindings;
			const auto dword      = layout.memory_offset_dword + index / 4u;
			const auto adjustment = dword < bindings.shader_data.size()
			                            ? (bindings.shader_data[dword] >> ((index % 4u) * 8u)) & 0xffu
			                            : 0u;
			host_start[index]  = info.offset + adjustment;
			host_stream[index] = info.buffer == stream;

			BufferProbeBinding entry;
			entry.index   = index;
			entry.guest   = source.address;
			entry.size    = source.size;
			entry.written = resource.written;
			entry.descriptor_size =
			    resource_index < snapshot.buffers.size()
			        ? DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource_index])
			              .GetSize()
			        : 0u;
			entry.state   = index < states.size() ? states[index] : BufferProbe::CacheState::Synced;
			entry.compared = std::min({source.size, MaxBytesPerBuffer, MaxBytesPerDispatch - total});
			if (entry.compared == 0) continue;
			entry.guest_bytes.resize(static_cast<size_t>(entry.compared));
			entry.guest_ok = LibKernel::Memory::TryReadSparseBacking(
			    entry.guest, entry.guest_bytes.data(), entry.compared);
			entry.staging_offset = total;
			copies.emplace_back(host_start[index], total, entry.compared);
			sources.push_back(info.buffer);
			total += Common::AlignUp(entry.compared, 16);
			probed.push_back(std::move(entry));
		}
		if (probed.empty()) return;

		auto staging = std::make_unique<Buffer>(context.GetGraphics(), context.GetCommandScheduler(),
		                                        MemoryUsage::Download, 0,
		                                        vk::BufferUsageFlagBits::eTransferDst, total);
		const auto native = command.Handle();
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eShaderWrite |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
		                       nullptr);
		for (size_t i = 0; i < copies.size(); i++) {
			native.copyBuffer(sources[i], staging->Handle(), 1, &copies[i]);
		}
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr,
		                       0, nullptr);
		const uint32_t dispatch = m_dispatch++;
		std::vector<std::string> alias_lines =
		    FindSplitAliases(hash, dispatch, probed, host_start, sources);
		// Debug option, correctness over speed: submit what is recorded so far (every upload and
		// earlier dispatch of the stream plus the copy) and wait for it, so the findings are
		// logged before this dispatch can hang the queue. The deferred form never ran when the
		// dispatch itself hung.
		auto& scheduler = context.GetCommandScheduler();
		scheduler.Wait(scheduler.CurrentTick());
		Report(hash, dispatch, total, host_stream, probed, alias_lines, *staging);
	}

private:
	// Two bindings of one dispatch whose guest ranges overlap must see the same bytes, so they
	// must be views of one host range: same buffer and the same offset-from-guest-address.
	static std::vector<std::string>
	FindSplitAliases(uint64_t hash, uint32_t dispatch, const std::vector<BufferProbeBinding>& probed,
	                 const std::vector<uint64_t>& host_start, const std::vector<vk::Buffer>& sources) {
		std::vector<std::string> lines;
		for (size_t a = 0; a < probed.size(); a++) {
			for (size_t b = a + 1; b < probed.size(); b++) {
				const auto& x = probed[a];
				const auto& y = probed[b];
				if (x.guest >= y.guest + y.size || y.guest >= x.guest + x.size) continue;
				const bool same_buffer = sources[a] == sources[b];
				const int64_t delta_x = static_cast<int64_t>(host_start[x.index]) -
				                        static_cast<int64_t>(x.guest);
				const int64_t delta_y = static_cast<int64_t>(host_start[y.index]) -
				                        static_cast<int64_t>(y.guest);
				if (same_buffer && delta_x == delta_y) continue;
				char text[320];
				std::snprintf(text, sizeof(text),
				              "[buffer-probe] SPLIT-ALIAS shader=0x%016" PRIx64
				              " dispatch=%u buf=%u (guest=0x%012" PRIx64 " size=0x%" PRIx64
				              " written=%d) and buf=%u (guest=0x%012" PRIx64 " size=0x%" PRIx64
				              " written=%d) overlap in guest memory but are bound to different host "
				              "ranges (%s): the shader's views of the shared bytes diverge",
				              hash, dispatch, x.index, x.guest, x.size, x.written ? 1 : 0, y.index,
				              y.guest, y.size, y.written ? 1 : 0,
				              same_buffer ? "same host buffer, different offsets"
				                          : "different host buffers");
				lines.emplace_back(text);
			}
		}
		return lines;
	}

	void Report(uint64_t hash, uint32_t dispatch, uint64_t total,
	            const std::vector<bool>& host_stream, std::vector<BufferProbeBinding>& probed,
	            const std::vector<std::string>& alias_lines, Buffer& staging) {
		if (!staging.IsCoherent()) staging.Invalidate(0, total);
		const auto mapped = staging.Mapped();
		std::lock_guard lock(m_mutex);
		bool            mismatch   = false;
		bool            unexpected = false;
		std::vector<std::string> lines;
		for (const auto& entry: probed) {
			if (mapped.empty()) continue;
			BufferProbe::Comparison comparison;
			if (entry.guest_ok) {
				comparison = BufferProbe::Compare(
				    entry.guest_bytes.data(), mapped.data() + entry.staging_offset, entry.compared);
			}
			mismatch   = mismatch || comparison.Differs();
			unexpected = unexpected || BufferProbe::IsUnexpectedDifference(comparison, entry.state);
			auto line = BufferProbe::FormatLine(hash, dispatch, entry.index, entry.guest,
			                                    entry.size, entry.written, entry.state, entry.guest_ok,
			                                    comparison);
			if (entry.index < host_stream.size() && host_stream[entry.index]) line += " src=stream";
			if (entry.compared < entry.size) line += " TRUNCATED";
			char declared[48];
			std::snprintf(declared, sizeof(declared), " descriptor_size=0x%" PRIx64 "%s",
			              entry.descriptor_size,
			              entry.descriptor_size != entry.size ? " CLAMPED" : "");
			line += declared;
			lines.push_back(std::move(line));
		}
		const bool aliased = !alias_lines.empty();
		if (m_limiter.OnDispatch(mismatch || aliased, unexpected || aliased)) {
			for (const auto& line: alias_lines) Log::WriteToConsoleAndLog((line + "\n").c_str());
			for (const auto& line: lines) Log::WriteToConsoleAndLog((line + "\n").c_str());
		}
		if (m_limiter.SummaryDue()) {
			Log::WriteToConsoleAndLog((m_limiter.Summary() + "\n").c_str());
		}
	}

	std::mutex            m_mutex;
	BufferProbe::Limiter  m_limiter;
	uint32_t              m_dispatch = 0;
};

BufferProbeRunner& GetBufferProbe() {
	static BufferProbeRunner runner;
	return runner;
}

} // namespace

static bool HasStorageWrites(const ShaderComputeInputInfo& input_info,
                             const PreparedBindings&       bindings) {
	const auto& images = input_info.stage.module->info.info.images;
	return bindings.shared_memory.buffer != nullptr || HasShaderBufferWrites(input_info.stage) ||
	       std::ranges::any_of(images, [](const auto& image) {
		       return image.written &&
			          image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
	       });
}

// Whether a dispatch may write memory through any path. A dispatch that only reads is not ordered
// after earlier dispatches that only read, so no write may be missed here.
static bool DispatchMayWriteMemory(const ShaderStageRuntime& stage, const PreparedBindings& bindings) {
	const auto& program = stage.module->info;
	const auto& info    = program.info;
	return bindings.shared_memory.buffer != nullptr || bindings.gds.buffer != nullptr ||
	       program.has_address_writes || info.uses_dma || info.uses_gds || info.function_scratch ||
	       HasShaderBufferWrites(stage) ||
	       std::ranges::any_of(info.buffers,
	                           [](const auto& resource) { return resource.written || resource.atomic; }) ||
	       std::ranges::any_of(info.images,
	                           [](const auto& resource) { return resource.written || resource.atomic; });
}

// The guest buffers a compute program names through descriptors, with how it uses them. Memory it
// reaches through pointers it computes itself is not listed.
static std::vector<SkipTaint::Use> ComputeBufferUses(const ShaderStageRuntime& stage) {
	std::vector<SkipTaint::Use> uses;
	const auto& metadata  = stage.module->info.info.buffers;
	const auto& snapshot  = stage.resources->buffers;
	for (uint32_t i = 0; i < metadata.size() && i < snapshot.size(); i++) {
		if (metadata[i].descriptor_index == UINT32_MAX) continue;
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot[i]);
		auto       size       = ShaderRecompiler::IR::BufferBindingSize(metadata[i], descriptor);
		if (descriptor.NumRecords() == UINT32_MAX) size = std::min<uint64_t>(size, 256u << 20u);
		if (descriptor.Base48() == 0 || size == 0) continue;
		uses.push_back({descriptor.Base48(), size, metadata[i].read || !metadata[i].written,
		                metadata[i].written});
	}
	for (const auto& table: stage.resources->indirect_buffer_tables) {
		if (table.root_resource >= metadata.size()) continue;
		const auto& meta = metadata[table.root_resource];
		if (!meta.written) continue;
		for (const auto& entry: table.candidates->entries) {
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(entry);
			auto       size       = ShaderRecompiler::IR::BufferBindingSize(meta, descriptor);
			if (descriptor.NumRecords() == UINT32_MAX) size = std::min<uint64_t>(size, 256u << 20u);
			if (descriptor.Base48() == 0 || size == 0) continue;
			uses.push_back({descriptor.Base48(), size, meta.read || !meta.written, meta.written});
		}
	}
	return uses;
}

static uint64_t SteadyNowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

static std::vector<uint64_t> WrittenAddresses(std::span<const SkipTaint::Use> uses) {
	std::vector<uint64_t> addrs;
	for (const auto& u: uses) {
		if (u.written && u.address != 0) {
			addrs.push_back(u.address);
		}
	}
	return addrs;
}

static uint64_t CapturedDispatchKey(const HW::ComputeShaderInfo& cs_regs, uint32_t gx, uint32_t gy,
                                    uint32_t gz, uint32_t mode, bool async_compute,
                                    uint64_t indirect_args, bool is_indirect) {
	uint64_t h = cs_regs.cs_regs.data_addr;
	const auto mix = [&h](uint64_t v) {
		h ^= v + 0x9e3779b9u + (h << 6u) + (h >> 2u);
	};
	mix((uint64_t {gx} << 32u) | gy);
	mix((uint64_t {gz} << 32u) | mode);
	mix(indirect_args ^ (uint64_t {async_compute ? 1u : 0u} << 1u) ^
	    uint64_t {is_indirect ? 1u : 0u});
	for (uint32_t i = 0; i < 16u; ++i) {
		mix(cs_regs.cs_user_sgpr.value[i]);
	}
	return h;
}

void RenderExecutor::TaintSkippedDispatch(bool record_wait_hole) {
	const auto now = SteadyNowMs();
	if (record_wait_hole) {
		FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::Wait,
		                             WrittenAddresses(m_pending_uses));
		// Mark this frame incomplete so the presenter holds the last complete frame instead of
		// presenting a hole while the skipped producer's shader finishes compiling.
		NoteDrawSkippedForShaders();
		if (m_pending_dispatch.has_value()) {
			const auto& cap = *m_pending_dispatch;
			const auto  key = CapturedDispatchKey(cap.cs_regs, cap.groups[0], cap.groups[1],
			                                      cap.groups[2], cap.mode, cap.async_compute,
			                                      cap.indirect_args, cap.is_indirect);
			m_skip_catchup.Enqueue(key, cap.cs_regs.cs_regs.data_addr, m_pending_uses, cap, now);
		}
	}
	m_skip_taint.Taint(m_pending_uses, now);
	m_pending_uses.clear();
	m_pending_dispatch.reset();
}

void RenderExecutor::DrainSkippedDispatchesLocked(CommandBuffer& buffer) {
	if (m_replaying_catchup || m_skip_catchup.Empty()) {
		return;
	}
	const auto now = SteadyNowMs();
	// A replay reads the dispatch's descriptors from guest memory as they are now. Long after the
	// skip that memory may have been reused, so entries older than --catchup-max-age-ms are dropped
	// instead of replayed.
	if (const uint64_t max_age_ms = Config::CatchupMaxAgeMs(); max_age_ms != 0) {
		(void)m_skip_catchup.TakeReady(now, [&](const auto& entry) {
			return now >= entry.skipped_at_ms && now - entry.skipped_at_ms > max_age_ms
			           ? SkipCatchUp::ProbeState::Failed
			           : SkipCatchUp::ProbeState::Pending;
		});
		if (m_skip_catchup.Empty()) {
			return;
		}
	}
	auto ready = m_skip_catchup.TakeReady(now, [this](const auto& entry) -> SkipCatchUp::ProbeState {
		const auto&            cap = entry.payload;
		ShaderComputeInputInfo probe_info {};
		probe_info.async_compute = cap.async_compute;
		if (cap.is_indirect) {
			probe_info.indirect_args_address = cap.indirect_args;
		} else {
			const bool use_threads =
			    (cap.mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
			probe_info.dispatch_thread_dimensions = use_threads;
			probe_info.workgroup_counts[0]        = cap.groups[0];
			probe_info.workgroup_counts[1]        = cap.groups[1];
			probe_info.workgroup_counts[2]        = cap.groups[2];
			if (use_threads) {
				const uint32_t gs[] = {cap.cs_regs.cs_regs.num_thread_x,
				                       cap.cs_regs.cs_regs.num_thread_y,
				                       cap.cs_regs.cs_regs.num_thread_z};
				for (uint32_t a = 0; a < 3u; ++a) {
					const auto sz = std::max(gs[a], 1u);
					probe_info.workgroup_counts[a] =
					    cap.groups[a] / sz + (cap.groups[a] % sz != 0u);
				}
			}
		}
		m_context.GetImageHeap().BeginDraw();
		bool pending = false;
		if (!m_context.GetPipelineCache().GetComputeProgram(cap.cs_regs, probe_info, &pending)) {
			return pending ? SkipCatchUp::ProbeState::Pending : SkipCatchUp::ProbeState::Failed;
		}
		if (!m_context.GetPipelineCache().EnsureComputePipeline(probe_info, pending)) {
			return pending ? SkipCatchUp::ProbeState::Pending : SkipCatchUp::ProbeState::Failed;
		}
		return SkipCatchUp::ProbeState::Ready;
	});
	if (ready.empty()) {
		return;
	}
	const auto saved_cs = buffer.GetShaders().GetCs();
	m_replaying_catchup = true;
	for (const auto& item: ready) {
		const auto& cap = item.payload;
		buffer.GetShaders().SetCsInfo(cap.cs_regs);
		if (cap.is_indirect) {
			(void)DispatchIndirectLocked(buffer, cap.indirect_args, cap.mode, cap.async_compute,
			                             /*allow_async=*/false);
		} else {
			(void)DispatchDirectLocked(buffer, cap.groups[0], cap.groups[1], cap.groups[2],
			                           cap.mode, cap.async_compute, /*allow_async=*/false,
			                           cap.indirect_args);
		}
		static std::atomic<uint32_t> replay_logs {0};
		if (replay_logs.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("[catchup] replayed skipped cs shader=0x%016" PRIx64
			     " pending=%zu total_replayed=%" PRIu64 "\n",
			     cap.cs_regs.cs_regs.data_addr, m_skip_catchup.Size(),
			     m_skip_catchup.GetStats().replayed);
		}
	}
	m_replaying_catchup = false;
	buffer.GetShaders().SetCsInfo(saved_cs);
}

// Before a ready dispatch records: true when it reads memory a skipped dispatch was to write. It is
// then skipped itself and its outputs are tainted. Otherwise what it writes is fresh again.
// Only with --async-compute-wait-ms; costs one empty() test while nothing was skipped.
static bool SkipOnTaintedInput(SkipTaint& taint, const ShaderStageRuntime& stage,
                               const std::vector<SkipTaint::Use>& extra_reads = {}) {
	if (Config::AsyncComputeWaitMs() == 0) return false;
	if (taint.Empty()) return false;
	auto uses = ComputeBufferUses(stage);
	// Memory the dispatch itself reads before the shader runs (indirect group counts).
	uses.insert(uses.end(), extra_reads.begin(), extra_reads.end());
	const auto now  = SteadyNowMs();
	if (taint.ReadsTainted(uses, now)) {
		taint.Taint(uses, now);
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("[skip] cs hash=0x%016" PRIx64 " reads a buffer a skipped dispatch was to write; "
			     "skipped too (tainted ranges=%zu)\n",
			     stage.module->info.shader_hash, taint.Size());
		}
		return true;
	}
	taint.ClearWritten(uses);
	return false;
}

// The guest words of `count` dwords from `vaddr` as `[dispatch]   <what> <index>: ...` lines;
// one "not readable" line when the range is not backed.
static void LogGuestWords(const char* what, uint64_t vaddr, size_t count, int64_t first_index) {
	uint32_t words[DispatchWindowDwords + 8] {};
	count = std::min(count, std::size(words));
	if (vaddr == 0 || !LibKernel::Memory::TryReadBacking(vaddr, words, count * sizeof(uint32_t))) {
		LOGF("[dispatch]   %s 0x%016" PRIx64 ": not readable\n", what, vaddr);
		return;
	}
	char prefix[64];
	std::snprintf(prefix, sizeof(prefix), "[dispatch]   %s ", what);
	LOGF("%s", FormatDwordWindow(prefix, words, count, first_index).c_str());
}

// Where a reported dispatch came from (dispatchOrigin.h): the submitting guest thread with its FP
// control state and last library calls, the 64 dwords before the packet and the packet, and for an
// indirect dispatch the words around its arguments.
static void LogDispatchOrigin(const DispatchOrigin* origin) {
	if (origin == nullptr) {
		LOGF("[dispatch]   origin: not known\n");
		return;
	}
	const auto* submitter = origin->submitter;
	LOGF("[dispatch]   origin: packet=0x%016" PRIx64 " args=0x%012" PRIx64
	     " args_gpu_written=%d submitter_thread=%d submitter_mxcsr=0x%04" PRIx32
	     " submitter_x87=0x%04" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(origin->packet), origin->args_addr, origin->args_gpu_written,
	     submitter != nullptr ? submitter->thread_id : -1,
	     submitter != nullptr ? submitter->mxcsr : 0u,
	     submitter != nullptr ? static_cast<uint32_t>(submitter->x87_cw) : 0u);
	LOGF("[dispatch]   submitter calls (oldest first): %s\n",
	     submitter != nullptr ? FormatSubmitterCalls(*submitter).c_str() : "not known");
	if (origin->packet != nullptr) {
		// The packet is 5 dwords (header, x, y, z, initiator) for a direct dispatch; index 0 = header.
		const auto packet = reinterpret_cast<uint64_t>(origin->packet);
		LogGuestWords("words", packet - DispatchWindowDwords * sizeof(uint32_t),
		              DispatchWindowDwords + 5, -static_cast<int64_t>(DispatchWindowDwords));
	}
	if (origin->args_addr != 0) {
		// index 0 = the x count
		LogGuestWords("args", origin->args_addr - 8 * sizeof(uint32_t), 24, -8);
	}
}

// A dispatch whose group counts the guest chose: names it in the log when it is large (default on,
// rate-limited per shader) and says whether it must be skipped because it is above the group limit
// (dispatchLimits.h). `args_addr` is the indirect argument address, 0 for a direct dispatch;
// `source` names where the counts came from.
static bool JudgeLargeDispatch(SkipTaint& taint, const ShaderStageRuntime& stage,
                               const HW::ComputeShaderInfo& cs, const uint32_t (&threads)[3],
                               const uint32_t (&groups)[3], uint32_t mode, uint64_t args_addr,
                               const char* source) {
	const uint64_t group_threads = uint64_t {std::max(threads[0], 1u)} * std::max(threads[1], 1u) *
	                               std::max(threads[2], 1u);
	const bool     may_skip      = DispatchArgsGuard::Enabled();
	const auto     verdict       = JudgeDispatch(groups[0], groups[1], groups[2], group_threads,
	                                             DispatchGroupLimit(), may_skip);
	if (verdict == DispatchVerdict::Run) return false;
	const uint64_t hash = stage.module->info.shader_hash;
	static DispatchReportLimiter limiter;
	if (limiter.Allow(hash)) {
		const auto& user  = cs.cs_user_sgpr.value;
		const int   count = FindCountDword(user, 16, groups[0], std::max(threads[0], 1u));
		bool        tainted_args  = false;
		bool        tainted_input = false;
		if (!taint.Empty()) {
			const auto now = SteadyNowMs();
			if (args_addr != 0) {
				tainted_args = taint.ReadsTainted(
				    {{args_addr, sizeof(vk::DispatchIndirectCommand), true, false}}, now);
			}
			tainted_input = taint.ReadsTainted(ComputeBufferUses(stage), now);
		}
		char count_text[96] = "none";
		if (count >= 0) {
			std::snprintf(count_text, sizeof(count_text), "user[%d]=0x%08" PRIx32 " (as float %g)",
			              count, user[count], static_cast<double>(DwordAsFloat(user[count])));
		}
		LOGF("[dispatch] cs hash=0x%016" PRIx64 " source=%s args=0x%012" PRIx64
		     " groups=0x%" PRIx32 ",0x%" PRIx32 ",0x%" PRIx32 " (%" PRIu64 " groups, %" PRIu64
		     " invocations) local=%ux%ux%u initiator=0x%08" PRIx32
		     " [thread_dims=%u partial=%u ordered_append=%u wave32=%u] count_from_dword=%s "
		     "tainted_input=%u tainted_args=%u action=%s limit=%" PRIu32 "\n",
		     hash, source, args_addr, groups[0], groups[1], groups[2],
		     DispatchGroups(groups[0], groups[1], groups[2]),
		     DispatchInvocations(groups[0], groups[1], groups[2], group_threads), threads[0],
		     threads[1], threads[2], mode, (mode >> 5u) & 1u, (mode >> 1u) & 1u, (mode >> 3u) & 1u,
		     (mode >> 15u) & 1u, count_text, tainted_input ? 1u : 0u, tainted_args ? 1u : 0u,
		     verdict == DispatchVerdict::Skip ? "skipped" : "run", DispatchGroupLimit());
		LogDispatchOrigin(t_dispatch_origin);
	}
	return verdict == DispatchVerdict::Skip;
}

namespace {
thread_local uint32_t g_dispatch_sample_countdown = Common::FrameCost::StepClock::InitialDelay;

struct DispatchStepSample {
	Common::FrameCost::StepClock clock;
	uint64_t                     program_ns  = 0;
	uint64_t                     pipeline_ns = 0;
	uint64_t                     bind_ns     = 0;
	uint64_t                     guard_ns    = 0;
	uint64_t                     commit_ns   = 0;
	uint64_t                     record_ns   = 0;

	DispatchStepSample() noexcept: clock(g_dispatch_sample_countdown) {}

	void Commit() noexcept {
		if (!clock.Active()) return;
		clock.AddTo(record_ns);
		using Common::FrameCost::Counter;
		Common::FrameCost::Add(Counter::DispatchStepSamples);
		Common::FrameCost::Add(Counter::DispatchStepProgramNs, program_ns);
		Common::FrameCost::Add(Counter::DispatchStepPipelineNs, pipeline_ns);
		Common::FrameCost::Add(Counter::DispatchStepBindNs, bind_ns);
		Common::FrameCost::Add(Counter::DispatchStepGuardNs, guard_ns);
		Common::FrameCost::Add(Counter::DispatchStepCommitNs, commit_ns);
		Common::FrameCost::Add(Counter::DispatchStepRecordNs, record_ns);
		clock.Complete();
	}
};
} // namespace

bool RenderExecutor::DispatchDirect(CommandBuffer& buffer, uint32_t thread_group_x,
                                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
                                    bool async_compute, bool allow_async, uint64_t indirect_args) {
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& sh_ctx = buffer.GetShaders();

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return true;
	}

	Common::FrameCost::ScopedTimer lock_wait(Common::FrameCost::Counter::LockWaitNs);
	Common::LockGuard              lock(m_context.GetMutex());
	lock_wait.Stop();
	if (!m_skip_catchup.Empty()) {
		DrainSkippedDispatchesLocked(buffer);
	}
	return DispatchDirectLocked(buffer, thread_group_x, thread_group_y, thread_group_z, mode,
	                            async_compute, allow_async, indirect_args);
}

bool RenderExecutor::DispatchDirectLocked(CommandBuffer& buffer, uint32_t thread_group_x,
                                          uint32_t thread_group_y, uint32_t thread_group_z,
                                          uint32_t mode, bool async_compute, bool allow_async,
                                          uint64_t indirect_args) {
	auto& sh_ctx = buffer.GetShaders();
	m_pending_uses.clear();
	m_pending_dispatch.reset();
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::NullShader);
		Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return true;
	}

	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS     = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto  make_capture = [&]() -> CapturedDispatch {
		return CapturedDispatch {
		    .cs_regs       = cs_regs,
		    .groups        = {thread_group_x, thread_group_y, thread_group_z},
		    .mode          = mode,
		    .async_compute = async_compute,
		    .indirect_args = indirect_args,
		    .is_indirect   = false,
		};
	};

	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0]        = thread_group_x;
	input_info.workgroup_counts[1]        = thread_group_y;
	input_info.workgroup_counts[2]        = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	DispatchStepSample sample;
	m_context.GetImageHeap().BeginDraw();
	bool pending = false;
	if (!m_context.GetPipelineCache().GetComputeProgram(cs_regs, input_info,
	                                                    allow_async ? &pending : nullptr)) {
		if (pending && Config::AsyncComputeWaitMs() != 0) {
			m_pending_dispatch = make_capture();
		} else if (!pending) {
			FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::FailedShader);
		}
		if (!pending) Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return !pending;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = input_info.stage.module->info;
	const auto& resources = *input_info.stage.resources;
	{
		uint32_t final_groups[3] = {thread_group_x, thread_group_y, thread_group_z};
		if (use_thread_dimensions) {
			std::copy(std::begin(input_info.workgroup_counts), std::end(input_info.workgroup_counts),
			          std::begin(final_groups));
		}
		if (JudgeLargeDispatch(m_skip_taint, input_info.stage, sh_ctx.GetCs(), input_info.threads_num,
		                       final_groups, mode, indirect_args,
		                       indirect_args != 0 ? "indirect-threads" : "direct")) {
			m_pending_uses = ComputeBufferUses(input_info.stage);
			FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::Refused,
			                             WrittenAddresses(m_pending_uses));
			TaintSkippedDispatch(/*record_wait_hole=*/false);
			Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
			return true;
		}
	}
	if (resources.specialization_reads.empty() &&
	    (TryConsumeComputeMetaClear(input_info, buffer) ||
	     TryConsumeComputeFill(input_info, buffer, thread_group_x, thread_group_y,
	                           thread_group_z, mode))) {
		ResetBindings();
		Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return true;
	}
	sample.clock.AddTo(sample.program_ns);
	if (allow_async && !m_context.GetPipelineCache().EnsureComputePipeline(input_info, pending)) {
		if (pending && Config::AsyncComputeWaitMs() != 0) {
			m_pending_uses     = ComputeBufferUses(input_info.stage);
			m_pending_dispatch = make_capture();
		} else if (!pending) {
			FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::PipelineFailed,
			                             WrittenAddresses(ComputeBufferUses(input_info.stage)));
		}
		if (!pending) Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return !pending;
	}
	// The thread counts of an indirect thread-dimension dispatch are an input too (read by the
	// command processor from the argument buffer): a skipped producer's counts are stale.
	if (allow_async &&
	    SkipOnTaintedInput(m_skip_taint, input_info.stage,
	                       indirect_args != 0
	                           ? std::vector<SkipTaint::Use> {{indirect_args,
	                                                           sizeof(vk::DispatchIndirectCommand),
	                                                           true, false}}
	                           : std::vector<SkipTaint::Use> {})) {
		const auto uses = ComputeBufferUses(input_info.stage);
		FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::Tainted,
		                             WrittenAddresses(uses));
		NoteDrawSkippedForShaders();
		const auto key = CapturedDispatchKey(cs_regs, thread_group_x, thread_group_y,
		                                     thread_group_z, mode, async_compute, indirect_args,
		                                     false);
		m_skip_catchup.Enqueue(key, cs_regs.cs_regs.data_addr, uses, make_capture(), SteadyNowMs());
		Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return true;
	}
	if (m_replaying_catchup && !m_skip_taint.Empty()) {
		m_skip_taint.ClearWritten(ComputeBufferUses(input_info.stage));
	}
	if (!m_replaying_catchup && !m_skip_catchup.Empty()) {
		const auto key = CapturedDispatchKey(cs_regs, thread_group_x, thread_group_y,
		                                     thread_group_z, mode, async_compute, indirect_args,
		                                     false);
		m_skip_catchup.DropByKey(key);
		m_skip_catchup.DropSuperseded(ComputeBufferUses(input_info.stage));
	}

	if (use_thread_dimensions) {
		thread_group_x = input_info.workgroup_counts[0];
		thread_group_y = input_info.workgroup_counts[1];
		thread_group_z = input_info.workgroup_counts[2];
	}

	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info);
	sample.clock.AddTo(sample.pipeline_ns);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	PreparedBindings* descriptor_stage = &bindings;
	CacheDmaBases(m_context, input_info.stage);
	FindBuffers(std::span {&descriptor_stage, 1u});
	const bool buffer_probe = Config::BufferProbeShader(program.shader_hash);
	std::vector<BufferProbe::CacheState> probe_states;
	if (buffer_probe) probe_states = BufferProbeRunner::SampleCacheState(m_context, bindings);
	if (program.info.uses_dma || program.info.uses_device_address) {
		m_context.PrepareBda(program.info.uses_dma);
	}
	PrepareRayQuery(m_context, input_info.stage, bindings);
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	BindSharedMemory(m_context, input_info, bindings);
	RebindBuffers(bindings);
	if (buffer_probe) {
		GetBufferProbe().Record(m_context, buffer, program.shader_hash, bindings, probe_states);
	}
	sample.clock.AddTo(sample.bind_ns);

	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	const bool tracked_barriers = Config::PortEnabled(Config::Port::BarrierTracker);
	const bool writes_memory    = HasStorageWrites(input_info, bindings) || program.info.uses_dma ||
	                           bindings.gds.buffer != nullptr;
	vk::CommandBuffer vk_buffer;
	if (tracked_barriers) {
		// The tracker orders this dispatch against the commands around it and defers the barrier
		// after it until a later command needs it.
		vk_buffer = buffer.DispatchHandle(DispatchMayWriteMemory(input_info.stage, bindings));
	} else {
		vk_buffer = buffer.Handle();
		if (writes_memory) {
			// A host fence used to serialize every dispatch. Preserve its read-before-write
			// ordering while allowing the queue to execute asynchronously.
			ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
	}
	sample.clock.AddTo(sample.commit_ns);
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			const auto  max_mip = image.r128 ? r.LastLevel() : r.MaxMip();
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(max_mip) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	auto&    graphics  = m_context.GetGraphics();
	uint32_t marker_id = 0;
	if (graphics.breadcrumbs.Active()) {
		OpBreadcrumb dispatch_op {};
		dispatch_op.kind        = BreadcrumbOpKind::Dispatch;
		dispatch_op.dim_x       = thread_group_x;
		dispatch_op.dim_y       = thread_group_y;
		dispatch_op.dim_z       = thread_group_z;
		dispatch_op.stage_count = 1;
		dispatch_op.stages[0]   = {program.stage, program.shader_hash, input_info.stage.module->id,
		                           program.info.uses_dma};
		CaptureComputeBufferBreadcrumbs(m_context, cs_regs, bindings, dispatch_op);
		marker_id = graphics.breadcrumbs.BeginOp(graphics, vk_buffer, dispatch_op);
	}
	constexpr uint32_t DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE = 1u << 3u;
	const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
	                                   std::max(input_info.threads_num[1], 1u) *
	                                   std::max(input_info.threads_num[2], 1u);
	if ((mode & DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE) != 0 && program.info.uses_gds &&
	    group_threads <= input_info.wave_size && !input_info.lds_storage) {
		const uint64_t total_groups =
		    uint64_t {thread_group_x} * uint64_t {thread_group_y} * uint64_t {thread_group_z};
		if (total_groups > 1u) {
			Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesOrderedAppendExtra,
			                       total_groups - 1u);
		}
		// A single guest wave per group can run in launch order without a device-wide
		// spinlock. Keep the guest workgroup IDs when splitting the dispatch.
		for (uint32_t z = 0; z < thread_group_z; ++z) {
			for (uint32_t y = 0; y < thread_group_y; ++y) {
				for (uint32_t x = 0; x < thread_group_x; ++x) {
					if ((x | y | z) != 0u) {
						if (tracked_barriers) {
							// A group sees the GDS and memory writes of the groups before it.
							const vk::MemoryBarrier order {
							    .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
							    .dstAccessMask = vk::AccessFlagBits::eShaderRead |
							                     vk::AccessFlagBits::eShaderWrite};
							vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
							                          vk::PipelineStageFlagBits::eComputeShader,
							                          vk::DependencyFlags {}, 1, &order, 0, nullptr,
							                          0, nullptr);
							Common::FrameCost::Add(Common::FrameCost::Counter::BarriersGlobal);
						} else {
							ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
						}
					}
					vk_buffer.dispatchBase(x, y, z, 1, 1, 1);
				}
			}
		}
	} else {
		vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
	}
	graphics.breadcrumbs.EndOp(graphics, vk_buffer, marker_id);
	m_context.GetBufferCache().RecordPendingWrites(program.has_address_writes ||
	                                               program.info.uses_dma);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	Common::FrameCost::Add(Common::FrameCost::Counter::Dispatches);
	if (program.info.bvh) {
		Common::FrameCost::Add(Common::FrameCost::Counter::BvhDispatches);
	}

	// The removed host fence also ordered read-only dispatches before later writers.
	if (tracked_barriers) {
		Common::FrameCost::Add(Common::FrameCost::Counter::BarriersDropped);
	} else {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader, writes_memory);
	}
	ResetBindings();
	sample.Commit();
	return true;
}

bool RenderExecutor::DispatchIndirect(CommandBuffer& buffer, uint64_t args_addr, uint32_t mode,
                                      bool async_compute, bool allow_async) {
	EXIT_IF(args_addr == 0 || (args_addr & 3u) != 0 ||
	        (!Config::PortEnabled(Config::Port::GpuIndirectThreadDim) &&
	         (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0));
	m_context.GetCommandScheduler().PopPendingOperations();
	Common::FrameCost::ScopedTimer lock_wait(Common::FrameCost::Counter::LockWaitNs);
	Common::LockGuard              lock(m_context.GetMutex());
	lock_wait.Stop();
	if (!m_skip_catchup.Empty()) {
		DrainSkippedDispatchesLocked(buffer);
	}
	return DispatchIndirectLocked(buffer, args_addr, mode, async_compute, allow_async);
}

bool RenderExecutor::DispatchIndirectLocked(CommandBuffer& buffer, uint64_t args_addr,
                                            uint32_t mode, bool async_compute, bool allow_async) {
	m_pending_uses.clear();
	m_pending_dispatch.reset();
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	vk::DispatchIndirectCommand args {};
	// The counts are known here only when the guest wrote them (the GPU-written ones are checked
	// by DispatchArgsGuard on the GPU).
	const bool args_gpu_dirty = m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args));
	const bool args_on_host   = !args_gpu_dirty &&
	                          LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args));
	if (args_on_host && (args.x == 0 || args.y == 0 || args.z == 0)) {
		Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return true;
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::NullShader);
		Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return true;
	}
	vk::DispatchIndirectCommand host_groups = args;
	if (args_on_host && use_thread_dimensions) {
		const uint32_t gx = std::max(cs_regs.cs_regs.num_thread_x, 1u);
		const uint32_t gy = std::max(cs_regs.cs_regs.num_thread_y, 1u);
		const uint32_t gz = std::max(cs_regs.cs_regs.num_thread_z, 1u);
		host_groups.x     = args.x / gx + (args.x % gx != 0u);
		host_groups.y     = args.y / gy + (args.y % gy != 0u);
		host_groups.z     = args.z / gz + (args.z % gz != 0u);
	}
	const auto make_capture = [&]() -> CapturedDispatch {
		return CapturedDispatch {
		    .cs_regs       = cs_regs,
		    .groups        = {0, 0, 0},
		    .mode          = mode,
		    .async_compute = async_compute,
		    .indirect_args = args_addr,
		    .is_indirect   = true,
		};
	};
	ShaderComputeInputInfo input_info {};
	input_info.async_compute              = async_compute;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.indirect_thread_dimensions = use_thread_dimensions;
	input_info.indirect_args_address      = args_addr;
	DispatchStepSample sample;
	m_context.GetImageHeap().BeginDraw();
	bool pending = false;
	if (!m_context.GetPipelineCache().GetComputeProgram(cs_regs, input_info,
	                                                    allow_async ? &pending : nullptr)) {
		if (pending && Config::AsyncComputeWaitMs() != 0) {
			m_pending_dispatch = make_capture();
		} else if (!pending) {
			FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::FailedShader);
		}
		if (!pending) Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return !pending;
	}
	sample.clock.AddTo(sample.program_ns);
	if (allow_async && !m_context.GetPipelineCache().EnsureComputePipeline(input_info, pending)) {
		if (pending && Config::AsyncComputeWaitMs() != 0) {
			m_pending_uses     = ComputeBufferUses(input_info.stage);
			m_pending_dispatch = make_capture();
		} else if (!pending) {
			FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::PipelineFailed,
			                             WrittenAddresses(ComputeBufferUses(input_info.stage)));
		}
		if (!pending) Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return !pending;
	}
	// The argument buffer is an input of the dispatch: a skipped producer's counts are stale.
	if (allow_async &&
	    SkipOnTaintedInput(m_skip_taint, input_info.stage,
	                       {{args_addr, sizeof(vk::DispatchIndirectCommand), true, false}})) {
		const auto uses = ComputeBufferUses(input_info.stage);
		FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::Tainted,
		                             WrittenAddresses(uses));
		NoteDrawSkippedForShaders();
		const auto key =
		    CapturedDispatchKey(cs_regs, 0, 0, 0, mode, async_compute, args_addr, true);
		m_skip_catchup.Enqueue(key, cs_regs.cs_regs.data_addr, uses, make_capture(), SteadyNowMs());
		Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
		return true;
	}
	if (args_on_host) {
		const uint32_t final_groups[3] = {host_groups.x, host_groups.y, host_groups.z};
		if (JudgeLargeDispatch(m_skip_taint, input_info.stage, cs_regs, input_info.threads_num,
		                       final_groups, mode, args_addr,
		                       use_thread_dimensions ? "indirect-threads" : "indirect")) {
			m_pending_uses = ComputeBufferUses(input_info.stage);
			FrameHoles::NoteDispatchSkip(FrameHoles::DispatchSkipReason::Refused,
			                             WrittenAddresses(m_pending_uses));
			TaintSkippedDispatch(/*record_wait_hole=*/false);
			Common::FrameCost::Add(Common::FrameCost::Counter::DispatchesSkipped);
			return true;
		}
	}
	if (m_replaying_catchup && !m_skip_taint.Empty()) {
		m_skip_taint.ClearWritten(ComputeBufferUses(input_info.stage));
	}
	if (!m_replaying_catchup && !m_skip_catchup.Empty()) {
		const auto key =
		    CapturedDispatchKey(cs_regs, 0, 0, 0, mode, async_compute, args_addr, true);
		m_skip_catchup.DropByKey(key);
		m_skip_catchup.DropSuperseded(ComputeBufferUses(input_info.stage));
	}
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info);
	sample.clock.AddTo(sample.pipeline_ns);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	PreparedBindings* descriptor_stage = &bindings;
	CacheDmaBases(m_context, input_info.stage);
	FindBuffers(std::span {&descriptor_stage, 1u});
	const auto& program = input_info.stage.module->info;
	const bool buffer_probe = Config::BufferProbeShader(program.shader_hash);
	std::vector<BufferProbe::CacheState> probe_states;
	if (buffer_probe) probe_states = BufferProbeRunner::SampleCacheState(m_context, bindings);
	if (program.info.uses_dma || program.info.uses_device_address) {
		m_context.PrepareBda(program.info.uses_dma);
	}
	PrepareRayQuery(m_context, input_info.stage, bindings);
	BindSharedMemory(m_context, input_info, bindings, args_addr,
	                 args_on_host ? &host_groups : nullptr);
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF((args_offset & 3u) != 0);
	const vk::Buffer args_vk_buffer = args_buffer->Handle();
	sample.clock.AddTo(sample.bind_ns);
	// Temporary workaround: counts the GPU wrote are checked on the GPU (dispatchGuard.h).
	// For thread-dimension indirect dispatches the same 1x1 pass converts thread counts to group
	// counts on the GPU so the CPU does not read them back.
	vk::Buffer indirect_buffer = args_vk_buffer;
	uint64_t   indirect_offset = args_offset;
	const bool guard_enabled   = DispatchArgsGuard::Enabled();
	if (use_thread_dimensions) {
		if (!m_dispatch_guard) {
			m_dispatch_guard = std::make_unique<DispatchArgsGuard>(
			    m_context.GetGraphics(), m_context.GetCommandScheduler());
		}
		const uint64_t stamp =
		    args_gpu_dirty ? m_context.GetBufferCache().WriteStamp(args_addr, sizeof(args)) : 0;
		std::tie(indirect_buffer, indirect_offset) = m_dispatch_guard->Record(
		    buffer, *args_buffer, args_offset, args_addr, stamp, cs_regs.cs_regs.num_thread_x,
		    cs_regs.cs_regs.num_thread_y, cs_regs.cs_regs.num_thread_z, true, guard_enabled);
		Common::FrameCost::Add(Common::FrameCost::Counter::DispatchIndirectThreadDimGpu);
	} else if (guard_enabled) {
		// Counts the CPU read are already checked against the limits here (the dispatch was skipped
		// above when too large): no GPU pass. Counts the GPU wrote are checked on the GPU, once
		// per write of the range.
		const auto& device_limits = m_context.GetGraphics().GetPhysicalDeviceProperties().limits;
		if (args_on_host &&
		    CountsSafeWithoutGuard(args.x, args.y, args.z, device_limits.maxComputeWorkGroupCount[0],
		                           device_limits.maxComputeWorkGroupCount[1],
		                           device_limits.maxComputeWorkGroupCount[2], DispatchGroupLimit())) {
			Common::FrameCost::Add(Common::FrameCost::Counter::GuardCpuChecks);
		} else {
			if (!m_dispatch_guard) {
				m_dispatch_guard = std::make_unique<DispatchArgsGuard>(
				    m_context.GetGraphics(), m_context.GetCommandScheduler());
			}
			const uint64_t stamp =
			    args_gpu_dirty ? m_context.GetBufferCache().WriteStamp(args_addr, sizeof(args)) : 0;
			std::tie(indirect_buffer, indirect_offset) =
			    m_dispatch_guard->Record(buffer, *args_buffer, args_offset, args_addr, stamp);
		}
	}
	sample.clock.AddTo(sample.guard_ns);
	RebindBuffers(bindings);
	if (use_thread_dimensions &&
	    program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		EXIT_IF(bindings.shader_data_buffer.buffer == nullptr);
		const auto vk_cmd = buffer.Handle();
		const uint64_t dst_offset =
		    bindings.shader_data_buffer.offset +
		    static_cast<uint64_t>(program.bindings.dispatch_thread_dword) * sizeof(uint32_t);
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = args_vk_buffer;
		before.offset              = args_offset;
		before.size                = sizeof(vk::DispatchIndirectCommand);
		vk_cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
		                       nullptr);
		const vk::BufferCopy copy {args_offset, dst_offset, sizeof(vk::DispatchIndirectCommand)};
		vk_cmd.copyBuffer(args_vk_buffer, bindings.shader_data_buffer.buffer, 1, &copy);
		vk::BufferMemoryBarrier after {};
		after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask       = vk::AccessFlagBits::eShaderRead;
		after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		after.buffer              = bindings.shader_data_buffer.buffer;
		after.offset              = dst_offset;
		after.size                = sizeof(vk::DispatchIndirectCommand);
		vk_cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eComputeShader, {}, 0, nullptr, 1, &after,
		                       0, nullptr);
	}
	if (buffer_probe) {
		GetBufferProbe().Record(m_context, buffer, program.shader_hash, bindings, probe_states);
	}
	sample.clock.AddTo(sample.bind_ns);
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	const bool tracked_barriers = Config::PortEnabled(Config::Port::BarrierTracker);
	const bool writes_memory    = HasStorageWrites(input_info, bindings) || program.info.uses_dma ||
	                           bindings.gds.buffer != nullptr;
	vk::CommandBuffer vk_buffer;
	if (tracked_barriers) {
		vk_buffer = buffer.DispatchHandle(DispatchMayWriteMemory(input_info.stage, bindings));
		// The barriers the tracker records leave out nothing the argument read needs, but the read
		// happens at the draw-indirect stage, which the compute-to-compute barrier does not name:
		// order every earlier shader and transfer write before it (the guard's included).
		const vk::MemoryBarrier arguments {
		    .srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
		    .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead};
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
		                              vk::PipelineStageFlagBits::eComputeShader |
		                              vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, vk::DependencyFlags {}, 1,
		                          &arguments, 0, nullptr, 0, nullptr);
		Common::FrameCost::Add(Common::FrameCost::Counter::BarriersGlobal);
	} else {
		vk_buffer = buffer.Handle();
		if (writes_memory) {
			ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
	}
	sample.clock.AddTo(sample.commit_ns);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	auto&    graphics  = m_context.GetGraphics();
	uint32_t marker_id = 0;
	if (graphics.breadcrumbs.Active()) {
		OpBreadcrumb dispatch_op {};
		dispatch_op.kind        = BreadcrumbOpKind::DispatchIndirect;
		dispatch_op.stage_count = 1;
		dispatch_op.stages[0]   = {program.stage, program.shader_hash, input_info.stage.module->id,
		                           program.info.uses_dma};
		CaptureComputeBufferBreadcrumbs(m_context, cs_regs, bindings, dispatch_op);
		marker_id = graphics.breadcrumbs.BeginOp(graphics, vk_buffer, dispatch_op);
	}
	vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
	graphics.breadcrumbs.EndOp(graphics, vk_buffer, marker_id);
	m_context.GetBufferCache().RecordPendingWrites(program.has_address_writes ||
	                                               program.info.uses_dma);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	Common::FrameCost::Add(Common::FrameCost::Counter::Dispatches);
	if (program.info.bvh) {
		Common::FrameCost::Add(Common::FrameCost::Counter::BvhDispatches);
	}
	if (tracked_barriers) {
		Common::FrameCost::Add(Common::FrameCost::Counter::BarriersDropped);
	} else {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader, writes_memory);
	}
	ResetBindings();
	sample.Commit();
	return true;
}

} // namespace Libs::Graphics
