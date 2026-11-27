#include "common/emulatorConfig.h"
#include "common/ports.h"

#include "common/assert.h"
#include "common/debugCounters.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace Config {

static std::unique_ptr<ConfigOptions> g_config;

// Leaves room for the guard's own `limit + 1` comparison constant.
static constexpr uint32_t MaxLoopGuardLimit = 0xfffffff0u;

void Initialize() {
	EXIT_IF(g_config != nullptr);

	g_config = std::make_unique<ConfigOptions>();
}

void Shutdown() {
	Common::DebugCounters::SetEnabled(false);
	g_config.reset();
}

void Load(const ConfigOptions& cfg) {
	EXIT_IF(g_config == nullptr);
	EXIT_IF(cfg.user_name.empty() || cfg.user_name.size() > MAX_USER_NAME_LENGTH);
	EXIT_IF(!IsConfiguredUserIdValid(cfg.user_id));
	EXIT_IF(cfg.controller_speaker_volume > 100 || cfg.controller_vibration_intensity > 100);
	EXIT_IF(cfg.keyboard_player > static_cast<uint32_t>(MAX_PLAYERS));

	*g_config = cfg;
	if (const char* env_skip = std::getenv("KYTY_SKIP_SHADERS");
	    env_skip != nullptr && env_skip[0] != '\0') {
		if (!ParseShaderHashList(env_skip, g_config->skip_shader_hashes)) {
			EXIT("invalid KYTY_SKIP_SHADERS value: %s\n", env_skip);
		}
	}
	auto& skip = g_config->skip_shader_hashes;
	std::ranges::sort(skip);
	skip.erase(std::unique(skip.begin(), skip.end()), skip.end());
	if (const char* env_fault = std::getenv("KYTY_GPU_FAULT_QUERY");
	    env_fault != nullptr && env_fault[0] != '\0' && std::strcmp(env_fault, "0") != 0) {
		g_config->gpu_fault_query_enabled = true;
	}
	if (const char* env_bda_validate = std::getenv("KYTY_BDA_TABLE_VALIDATE");
	    env_bda_validate != nullptr && env_bda_validate[0] != '\0' &&
	    std::strcmp(env_bda_validate, "0") != 0) {
		g_config->bda_table_validate_enabled = true;
	}
	if (const char* env_probe = std::getenv("KYTY_BUFFER_PROBE");
	    env_probe != nullptr && env_probe[0] != '\0') {
		if (!ParseShaderHashList(env_probe, g_config->buffer_probe_hashes)) {
			EXIT("invalid KYTY_BUFFER_PROBE value: %s\n", env_probe);
		}
	}
	auto& probe = g_config->buffer_probe_hashes;
	std::ranges::sort(probe);
	probe.erase(std::unique(probe.begin(), probe.end()), probe.end());
	if (const char* env_guard = std::getenv("KYTY_LOOP_GUARD");
	    env_guard != nullptr && env_guard[0] != '\0') {
		char*      end   = nullptr;
		const auto limit = std::strtoull(env_guard, &end, 0);
		if (end == env_guard || *end != '\0') {
			EXIT("invalid KYTY_LOOP_GUARD value: %s\n", env_guard);
		}
		g_config->loop_guard_limit = static_cast<uint32_t>(std::min<unsigned long long>(limit, MaxLoopGuardLimit));
		g_config->loop_guard_explicit = true;
	}
	g_config->loop_guard_limit = std::min<uint32_t>(g_config->loop_guard_limit, MaxLoopGuardLimit);
	SetDisabledPorts(g_config->disabled_ports);
	SetEnabledPorts(g_config->enabled_ports, g_config->disabled_ports);
	Common::DebugCounters::SetEnabled(cfg.perf_overlay_enabled);
}

bool ParseTraceCallsOption(std::string_view arg, ConfigOptions& cfg) {
	if (arg == "--trace-calls") {
		cfg.trace_calls_enabled = true;
		cfg.trace_calls_filter.clear();
		return true;
	}
	constexpr std::string_view prefix = "--trace-calls=";
	if (!arg.starts_with(prefix)) {
		return false;
	}
	const auto rest = arg.substr(prefix.size());
	if (rest.empty()) {
		return false;
	}
	std::vector<std::string> libs;
	size_t                   pos = 0;
	while (pos < rest.size()) {
		const size_t comma = rest.find(',', pos);
		const auto   token =
		    comma == std::string_view::npos ? rest.substr(pos) : rest.substr(pos, comma - pos);
		if (!token.empty()) {
			libs.emplace_back(token);
		}
		if (comma == std::string_view::npos) {
			break;
		}
		pos = comma + 1;
	}
	if (libs.empty()) {
		return false;
	}
	cfg.trace_calls_enabled = true;
	cfg.trace_calls_filter  = std::move(libs);
	return true;
}

uint32_t GetScreenWidth() {
	return g_config->screen_width;
}

uint32_t GetScreenHeight() {
	return g_config->screen_height;
}

const std::string& GetUserName() {
	return g_config->user_name;
}

int32_t GetUserId() {
	return g_config->user_id;
}

const std::string& GetAudioInputDevice() {
	return g_config->audio_input_device;
}

AudioOutput GetAudioOutput() {
	return g_config->audio_output;
}

bool Audio3dEnabled() {
	return g_config->audio_3d_enabled;
}

const std::optional<ControllerColor>& GetControllerColor() {
	return g_config->controller_color;
}

uint32_t GetControllerSpeakerVolume() {
	return g_config->controller_speaker_volume;
}

uint32_t GetControllerVibrationIntensity() {
	return g_config->controller_vibration_intensity;
}

uint32_t GetKeyboardPlayer() {
	return g_config->keyboard_player;
}

int32_t GetPlayerUserId(int player) {
	return g_config->user_id + player * PLAYER_USER_ID_STEP;
}

int GetPlayerFromUserId(int32_t user_id) {
	for (int player = 0; player < MAX_PLAYERS; player++) {
		if (user_id == GetPlayerUserId(player)) {
			return player;
		}
	}
	return -1;
}

bool IsGuestUserId(int32_t user_id) {
	return GetPlayerFromUserId(user_id) > 0;
}

PresentMode GetPresentMode() {
	return g_config->present_mode;
}

int32_t GetGpuIndex() {
	return g_config->gpu_index;
}

bool FullscreenEnabled() {
	return g_config->fullscreen_enabled;
}

bool HdrEnabled() {
	return g_config->hdr_enabled;
}

bool HideCursorEnabled() {
	return g_config->hide_cursor_enabled;
}

bool StretchEnabled() {
	return g_config->stretch_enabled;
}

bool VrEnabled() {
	return g_config->vr_enabled;
}

bool Ps5ProEnabled() {
	return g_config->pro_enabled;
}

PssrUpscaler GetPssrUpscaler() {
	return g_config->pssr_upscaler;
}

const std::filesystem::path& GetDlssFolder() {
	return g_config->dlss_folder;
}

uint32_t GetDlssPreset() {
	return g_config->dlss_preset;
}

const std::filesystem::path& GetFsrFolder() {
	return g_config->fsr_folder;
}

bool AmdCpuEnabled() {
	return g_config->amd_cpu_enabled;
}

bool GuestTscEnabled() {
	return g_config->guest_tsc_enabled;
}

uint32_t GetVblankFrequency() {
	return std::clamp(g_config->vblank_frequency, 30u, 360u);
}

uint32_t GetConsoleLanguage() {
	return g_config->console_language;
}

bool VulkanValidationEnabled() {
	return g_config->vulkan_validation_enabled;
}

bool IsNetworkConnected() {
	return g_config->network_connected;
}

bool PsnEmulationEnabled() {
	return g_config->psn_emulation_enabled;
}

bool ShaderValidationEnabled() {
	return g_config->shader_validation_enabled;
}

bool KeepGoingEnabled() {
	return g_config->keep_going_enabled;
}

int32_t AutoCrossSeconds() {
	return g_config->auto_cross_seconds;
}

bool TraceCallsEnabled() {
	return g_config->trace_calls_enabled;
}

const std::vector<std::string>& TraceCallsFilter() {
	return g_config->trace_calls_filter;
}

bool ParseShaderHashList(std::string_view text, std::vector<uint64_t>& out) {
	std::vector<uint64_t> parsed;
	size_t                pos = 0;
	while (pos < text.size()) {
		while (pos < text.size()) {
			const char c = text[pos];
			if (c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\r' || c == '\n') {
				++pos;
				continue;
			}
			break;
		}
		if (pos >= text.size()) {
			break;
		}
		if (text[pos] == '#') {
			while (pos < text.size() && text[pos] != '\n') {
				++pos;
			}
			continue;
		}
		const size_t start = pos;
		while (pos < text.size()) {
			const char c = text[pos];
			if (c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\r' || c == '\n' ||
			    c == '#') {
				break;
			}
			++pos;
		}
		std::string_view token = text.substr(start, pos - start);
		if (token.starts_with("0x") || token.starts_with("0X")) {
			token.remove_prefix(2);
		}
		if (token.empty() || token.size() > 16) {
			return false;
		}
		uint64_t value = 0;
		auto [end, ec] = std::from_chars(token.data(), token.data() + token.size(), value, 16);
		if (ec != std::errc {} || end != token.data() + token.size()) {
			return false;
		}
		parsed.push_back(value);
	}
	out.insert(out.end(), parsed.begin(), parsed.end());
	return true;
}

bool ShouldSkipShader(uint64_t hash) {
	if (g_config == nullptr) {
		return false;
	}
	const auto& hashes = g_config->skip_shader_hashes;
	return !hashes.empty() && std::ranges::binary_search(hashes, hash);
}

const std::vector<uint64_t>& GetSkipShaderHashes() {
	return g_config->skip_shader_hashes;
}

bool ShaderCacheEnabled() {
	return g_config->shader_cache_enabled;
}

bool ShaderCacheKeepEnabled() {
	return g_config->shader_cache_keep_enabled;
}

bool ShaderCacheRetryFailed() {
	return g_config->shader_cache_retry_failed;
}

bool AsyncShaderCompileEnabled() {
	return g_config->async_shader_compile_enabled;
}

LogDirection GetShaderLogDirection() {
	return g_config->shader_log_direction;
}

std::filesystem::path GetShaderLogFolder() {
	return g_config->shader_log_folder;
}

bool CommandBufferDumpEnabled() {
	return g_config->command_buffer_dump_enabled;
}

std::filesystem::path GetCommandBufferDumpFolder() {
	return g_config->command_buffer_dump_folder;
}

bool GraphicsDebugDumpEnabled() {
	return g_config->graphics_debug_dump_enabled;
}

LogDirection GetPrintfDirection() {
	return g_config->printf_direction;
}

std::filesystem::path GetPrintfOutputFile() {
	return g_config->printf_output_file;
}

bool ProfilerEnabled() {
	return g_config->profiler_enabled;
}

bool SpirvDebugPrintfEnabled() {
	return g_config->spirv_debug_printf_enabled;
}

bool GpuAssistedValidationEnabled() {
	return g_config->gpu_assisted_validation_enabled && g_config->vulkan_validation_enabled;
}

bool RenderDocEnabled() {
	return g_config->renderdoc_enabled;
}

bool ReadbackLinearImagesEnabled() {
	return g_config->readback_linear_images;
}

bool ReadbackTiledImagesEnabled() {
	return g_config->readback_tiled_images;
}

bool ReadbackStorageImagesEnabled() {
	return g_config->readback_storage_images;
}

bool SyncRawImageBuffersEnabled() {
	return g_config->sync_raw_image_buffers;
}

uint32_t GetFlushDraws() {
	return g_config->flush_draws;
}

bool TessellationEnabled() {
	return g_config->tessellation_enabled;
}

bool OcclusionQueriesEnabled() {
	return g_config->occlusion_queries_enabled;
}

bool RelaxedShaderFloatEnabled() {
	return g_config->relaxed_shader_float;
}

bool NativeRtEnabled() {
	return g_config->native_rt;
}

bool LodStatsEnabled() {
	return g_config->lod_stats_enabled;
}

bool TrophyEnabled() {
	return g_config->trophy_enabled;
}

uint32_t GetTrophyDurationSeconds() {
	return std::clamp(g_config->trophy_duration_seconds, 5u, 10u);
}

bool PlayGoHackEnabled() {
	return g_config->playgo_hack_enabled;
}

bool SkipNoticeScreen() {
	return g_config->skip_notice_screen;
}

bool GpuFaultQueryEnabled() {
	return g_config->gpu_fault_query_enabled;
}

const std::string& GetLaunchActivity() {
	return g_config->launch_activity;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled() {
	return g_config->red_zone_protection_enabled;
}
#endif

const Keymap& GetKeymap() {
	return g_config->keymap;
}

const std::vector<std::string>& GetGuestArguments() {
	return g_config->guest_arguments;
}

bool BdaTableValidateEnabled() {
	return g_config != nullptr && g_config->bda_table_validate_enabled;
}

bool BufferProbeShader(uint64_t hash) {
	return g_config != nullptr && !g_config->buffer_probe_hashes.empty() &&
	       std::ranges::binary_search(g_config->buffer_probe_hashes, hash);
}

const std::vector<uint64_t>& BufferProbeHashes() {
	static const std::vector<uint64_t> none;
	return g_config != nullptr ? g_config->buffer_probe_hashes : none;
}

bool BufferProbeEnabled() {
	return g_config != nullptr && !g_config->buffer_probe_hashes.empty();
}

uint32_t LoopGuardLimit() {
	return g_config != nullptr ? g_config->loop_guard_limit : 0u;
}

uint32_t ComputeLoopGuardLimit() {
	if (g_config == nullptr) return 0u;
	return g_config->loop_guard_explicit ? g_config->loop_guard_limit : DefaultComputeLoopGuardLimit;
}

uint32_t HangDumpWaitSeconds() {
	return g_config != nullptr ? g_config->hang_dump_wait_seconds : 60u;
}

bool ShaderKeyIndexEnabled() {
	return g_config != nullptr && g_config->shader_key_index_enabled;
}

uint32_t AsyncPumpBudgetMs() {
	return g_config != nullptr ? g_config->async_pump_budget_ms : 2u;
}

uint32_t AsyncWorkers() {
	return g_config != nullptr ? g_config->async_workers : 0u;
}

uint32_t AsyncSpeculativeModulesMode() {
	return g_config != nullptr ? g_config->async_speculative_modules : 1u;
}

uint32_t AsyncComputeWaitMs() {
	return g_config != nullptr ? g_config->async_compute_wait_ms : 0u;
}

uint32_t MemLimitGb() {
	return g_config != nullptr ? g_config->mem_limit_gb : 0u;
}

uint32_t FaultStormFaults() {
	return g_config != nullptr ? g_config->fault_storm_faults : 0u;
}

uint32_t FaultStormFrames() {
	return g_config != nullptr ? g_config->fault_storm_frames : 8u;
}

uint32_t FaultStormHold() {
	return g_config != nullptr ? g_config->fault_storm_hold : 60u;
}

bool AsyncLookaheadEnabled() {
	return g_config == nullptr || g_config->async_lookahead != 0u;
}

bool AsyncSpeculativeModules() {
	return g_config == nullptr || g_config->async_speculative_modules != 0u;
}

uint32_t RenderStallReportSeconds() {
	return g_config != nullptr ? g_config->render_stall_report_s : 5u;
}

uint32_t AsyncIrBudgetMb() {
	return g_config != nullptr ? g_config->async_ir_budget_mb : 2048u;
}

bool GuestFaultParkEnabled() {
	return g_config != nullptr && g_config->guest_fault_park;
}

bool GuestFaultReportOnlyEnabled() {
	return g_config != nullptr && g_config->guest_fault_report_only;
}

uint32_t GuestClockHoldMs() {
	return g_config != nullptr ? g_config->guest_clock_hold_ms : 0u;
}

const std::filesystem::path& GetRegisteredShaderDumpFolder() {
	static const std::filesystem::path none;
	return g_config != nullptr ? g_config->dump_registered_shaders_folder : none;
}

uint32_t WarmUpBudgetSeconds() {
	return g_config != nullptr ? g_config->warmup_budget_s : 0u;
}

bool SpirvCompactEnabled() {
	return g_config != nullptr && g_config->spirv_compact_enabled;
}

uint32_t MaterializeMemoEntries() {
	return g_config != nullptr ? g_config->materialize_memo_entries : 1u;
}

uint32_t MaterializeMemoStamps() {
	return g_config != nullptr ? g_config->materialize_memo_stamps : 0u;
}

uint32_t MaxSpirvWords() {
	return g_config != nullptr ? g_config->max_spirv_words : ConfigOptions::DefaultMaxSpirvWords;
}

uint32_t HeapTuneArenas() {
	return g_config != nullptr ? g_config->heap_tune_arenas : 0u;
}

uint32_t DriverCacheCapMb() {
	return g_config != nullptr ? g_config->driver_cache_cap_mb : 0u;
}

bool GpuDebugEnabled() {
	return g_config != nullptr && g_config->gpu_debug_enabled;
}

uint32_t PrefetchBackoffExponent() {
	return g_config != nullptr ? std::min(g_config->prefetch_backoff, 16u) : 0u;
}

uint32_t FrameTimerSample() {
	return g_config != nullptr ? std::max(g_config->frame_timer_sample, 1u) : 1u;
}

uint32_t DeferNewTargetsOnStallMs() {
	return g_config != nullptr ? g_config->defer_new_targets_on_stall_ms : 0u;
}

uint32_t CatchupMaxAgeMs() {
	return g_config != nullptr ? g_config->catchup_max_age_ms : 0u;
}

bool RuntimeTableMark() {
	return g_config == nullptr || g_config->runtime_table_mark != 0;
}

} // namespace Config
