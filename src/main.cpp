#include "common/archive.h"
#include "common/common.h"
#include "common/file.h"
#include "common/ports.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "emulator.h"
#include "kytyGitVersion.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>
#include <magic_enum.hpp>

using namespace Common;
using namespace Emulator;

static void PrintUsage() {
	::printf("%s, %s, ver = %s, git = %s, date = %s\n",
	         KYTY_BUILD == KYTY_BUILD_DEBUG ? "Debug" : "Release", KYTY_TOOLCHAIN, KYTY_VERSION,
	         KYTY_GIT_VERSION, __DATE__);
	::printf("kyty_emulator --game <dir|elf|zar> [options]\n\n");
	::printf("Options:\n");
	::printf("  --game <dir|elf|zar>                 Game directory, ELF, or ZArchive to load.\n");
	::printf("  --game-patch <json>                  ETAHen cheat file.\n");
	::printf("  --addcont-dir <dir>                  Add-ons (DLC) folder (default: none).\n");
	::printf(
	    "  --addcont <label>[=<dir>]            Add-on (DLC) by entitlement label with its data in\n"
	    "                                       <dir>, or license only without <dir>. Repeatable.\n");
	::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
	::printf("  --screen-height <num>                Window height. Default: 720.\n");
	::printf(
	    "  --user-name <name>                   Local user name (1-16 bytes). Default: Kyty.\n");
	::printf("  --user-id <num>                      Local user ID. Default: %d.\n",
	         Config::DEFAULT_USER_ID);
	::printf("  --mic <name>                        Capture from this microphone; omit for silence.\n");
	::printf("  --audio-3d                          Enable 3D audio support for games.\n");
	::printf("  --audio-output <value>              Report TV, AVReceiver (7.1), SoundBar (5.1),\n"
	         "                                      or Headphones to games. Default: TV.\n");
	::printf("  --controller-color <#RRGGBB>        Override the controller lightbar color.\n");
	::printf("  --controller-volume <0-100>         DualSense speaker volume. Default: 50.\n");
	::printf("  --controller-vibration <0-100>      DualSense vibration intensity. Default: 100.\n");
	::printf("  --keyboard-player <0-4>             Keyboard player; 0 shares player 1 with the gamepad. Default: 0.\n");
	::printf(
	    "  --present-mode <value>               Fifo, Mailbox, or Immediate. Default: Mailbox.\n");
	::printf(
	    "  --gpu <index>                        Vulkan physical device index. Default: auto.\n");
	::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
	::printf("  --hdr                                Enable HDR10 output when supported.\n");
	::printf(
	    "  --hide-cursor                        Hide the cursor after 2 s idle. Default: off.\n");
	::printf(
	    "  --stretch                            Stretch the image to the window. Default: off.\n");
	::printf("  --no-splash                          Disable the startup splash screen.\n");
	::printf("  --vr                                 Enable the virtual VR headset.\n");
	::printf("  --trinity                            Enable experimental PS5 Pro mode for\n"
	         "                                       supported titles. Default: off.\n");
	::printf("  --pssr-upscaler <value>              With --trinity, upscale PSSR with Linear,\n"
	         "                                       DLSS, or FSR. Default: Linear.\n");
	::printf("  --dlss-folder <folder>               DLSS model folder.\n");
	::printf("  --dlss-preset <A-O>                  DLSS preset. Default: the model's own.\n");
	::printf("  --fsr-folder <folder>                FidelityFX DLL folder (Windows).\n");
	::printf("  --amd-cpu                            Apply AMD CPU instruction patches.\n");
	::printf("  --guest-tsc                          Run the guest TSC at the PS5's 1.6 GHz.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
	::printf("  --vulkan-validation                  Enable Vulkan validation.\n");
	::printf("  --network-connected                  Use the host local network.\n");
	::printf("  --emulate-psn                        Simulate PSN sign-in for compatibility.\n");
	::printf("  --gpu-assisted-validation            Bounds-check shader accesses on the GPU.\n"
	         "                                       Implies --vulkan-validation; very slow.\n");
	::printf("  --shader-validation                  Enable shader validation.\n");
	::printf("  --auto-cross <seconds>               Debug: for unattended runs, report Cross pressed for 0.15 s\n"
	         "                                       every 2 s for <seconds> from the title's first pad read.\n");
	::printf("  --trace-calls[=<lib[,lib]>]          Debug: log system library calls (1st-4th and powers\n"
	         "                                       of two; with <lib[,lib]>, log every call with args/result).\n");
	::printf("  --keep-going                         Debug: skip draws and dispatches whose shader fails to\n"
	         "                                       compile, record them in _ShaderFailures/, keep running.\n");
	::printf("  --skip-shaders <hashes|@file>        Debug: skip draws and dispatches matching any comma-\n"
	         "                                       separated hex shader hash (or @path). Repeatable; also\n"
	         "                                       reads KYTY_SKIP_SHADERS.\n");
	::printf("  --gpu-fault-query                    Debug: query VK_EXT_device_fault after flushing\n"
	         "                                       device-lost breadcrumbs (also KYTY_GPU_FAULT_QUERY=1).\n");
	::printf("  --gpu-debug                          Debug: record a breadcrumb and a driver marker for every draw\n"
	         "                                       and dispatch (device-lost reports name the operation).\n"
	         "                                       Also on with --gpu-fault-query, --graphics-debug-dump,\n"
	         "                                       --guest-fault-report-only or RADV_DEBUG containing hang.\n");
	::printf("  --bda-table-validate                 Debug: validate every BDA page-table write and clear\n"
	         "                                       against live buffers (also KYTY_BDA_TABLE_VALIDATE=1).\n");
	::printf("  --shader-cache                       Keep compiled shaders and pipelines between boots.\n");
	::printf("  --shader-cache-keep                  Keep usable records from a mismatched shader cache,\n"
	         "                                       dropping only records that fail to load.\n");
	::printf("  --shader-cache-retry-failed          Ignore negative-cache entries in the shader store,\n"
	         "                                       forcing failed shaders to retranslate.\n");
	::printf("  --no-async-shader-compile            Compile shaders on the render thread, which waits for\n"
	         "                                       them. By default they compile on worker threads and\n"
	         "                                       draws using a shader still being compiled are skipped.\n");
	::printf("  --tessellation                      Draw tessellation patches; skipped by default.\n");
	::printf("  --occlusion-queries                  Count occlusion query samples on the GPU.\n");
	::printf("  --relaxed-shader-float               Native GPU float ops in shaders; faster, can be inaccurate.\n");
	::printf("  --native-rt                          Use native RT on the GPU.\n");
	::printf("  --lod-stats                          Report bound textures in mip statistics.\n");
	::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
	::printf("  --shader-log-folder <path>           Shader log output folder.\n");
	::printf("  --command-buffer-dump                Enable command buffer dumps.\n");
	::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
	::printf("  --graphics-debug-dump                Enable graphics debug dumps.\n");
	::printf("  --printf-direction <value>           Silent, Console, or File.\n");
	::printf("  --printf-output-file <path>          Guest printf output file.\n");
	::printf("  --profile                            Enable the Tracy profiler.\n");
	::printf("  --spirv-debug-printf                 Enable SPIR-V debug printf.\n");
	::printf("  --readback-linear-images             Read back writable linear images on submit.\n");
	::printf("  --readback-tiled-images              Read back small writable tiled images on submit.\n");
	::printf("  --readback-storage-images            Read back storage images on GL2 writeback.\n");
	::printf("  --sync-raw-image-buffers             Synchronize raw reads of GPU images.\n");
	::printf(
	    "  --flush-draws <num>                  Submit GPU work every <num> draws; 0 is off.\n");
	::printf("  --no-trophy-notifications            Disable trophy unlock toasts and their sound.\n");
	::printf(
	    "  --trophy-duration <seconds>          Trophy toast duration, 5-10. Default: 6.\n");
	::printf("  --perf-overlay                       Show the performance panel (F2 or Cmd+P).\n");
	::printf("  --playgo-hack                        Report 1000 PlayGo chunks without chunk metadata.\n");
	::printf("  --skip-notice-screen                 Skip startup logos and notices in supported games.\n");
	::printf("  --launch-activity <id>               Launch into this activity (1-32 printable ASCII chars).\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::printf("  --redzone                            Protect the guest SysV red zone.\n");
#endif
	::printf("  --keymap <Control=Input>             DualSense mapping; may be repeated.\n");
	::printf("  --rd                                 Enable RenderDoc capture.\n");
	::printf("  --hang-dump-wait <seconds>           Debug: after a device loss, wait up to this long for the\n"
	         "                                       driver's own hang dump when RADV_DEBUG contains `hang`.\n"
	         "                                       0 = do not wait. Default: 60.\n");
	::printf("  --buffer-probe <hashes>              Debug: before each dispatch of a listed shader, compare the guest\n"
	         "                                       memory of every bound buffer with the host copy the GPU reads\n"
	         "                                       and log the differences. Also reads KYTY_BUFFER_PROBE.\n");
	::printf("  --loop-guard <n>                     After n evaluated loop exits an invocation leaves all its\n"
	         "                                       loops and the shader hash and loop are logged. Given, it\n"
	         "                                       applies to every shader; 0 = off. Default: compute shaders\n"
	         "                                       only, n = 1048576. Also reads KYTY_LOOP_GUARD.\n");
	::printf("  --disable-port <names>               Turn off ported fixes (comma-separated names, or all):\n");
	for (const auto& port: Config::PORTS) {
		::printf("                                         %s (%s)\n", port.name, port.source);
	}
	::printf("  --shader-key-index                   Persist a shader hash -> ProgramKey index in the\n"
	         "                                       shader store for next-boot lookup. Default: off.\n");
	::printf("  --async-pump-budget-ms <ms>          Render-thread time per frame step spent applying finished\n"
	         "                                       async shader jobs; 0 = unlimited. Default: 2.\n"
	         "  --async-workers <n>                  Shader worker threads. 0 = hardware threads minus 4, at\n"
	         "                                       least 2. Default: 0.\n"
	         "  --async-speculative-modules <0|1|2>  Build the module of a compute translation on the worker\n"
	         "                                       when its resources need no guest memory, so it is stored at\n"
	         "                                       once. 1 = only with the user data of a dispatch or of the\n"
	         "                                       command scan; 2 = also for registration prefetches, which\n"
	         "                                       have none (old behaviour). Default: 1.\n"
	         "  --render-stall-report-s <n>          When no frame was recorded for n seconds, print one line\n"
	         "                                       with what the render thread is doing, then every n\n"
	         "                                       seconds. 0 = off. Default: 5.\n"
	         "  --async-ir-budget-mb <n>             Memory finished shader translations that no draw has\n"
	         "                                       asked for yet may hold; over it the least recently\n"
	         "                                       requested are dropped. 0 = no limit. Default: 2048.\n"
	         "  --guest-fault-park                   Debug: a guest thread that takes a fault nobody resolves\n"
	         "                                       prints the fault report once per pc and parks forever\n"
	         "                                       instead of ending the process.\n"
	         "  --guest-fault-report-only            Debug: on that fault also write fault-code-<n>.txt (hex of\n"
	         "                                       0x400 guest code bytes around the pc and each return\n"
	         "                                       address) into $KYTY_RUN_OUT, or the current folder.\n"
	         "  --guest-clock-hold-on-stall <ms>     Debug: the guest clock (time stamp counter, process time,\n"
	         "                                       clock_gettime, gettimeofday) stands still while no frame\n"
	         "                                       was recorded for more than <ms> and the render side is\n"
	         "                                       busy with shaders. 0 = off. Default: 0.\n"
	         "  --dump-registered-shaders <dir>      Write each distinct registered shader and index.jsonl\n"
	         "                                       through a bounded background queue. Default: off.\n"
	         "  --async-compute-wait-ms <n>          Bring-up: a compute dispatch whose shader is not ready\n"
	         "                                       suspends its queue at most n ms, then is skipped for this\n"
	         "                                       use (its outputs stay as they were) while the build goes\n"
	         "                                       on. 0 = wait for ever. Default: 0.\n"
	         "  --async-lookahead <0|1>              The scan of submitted commands also reads compute user\n"
	         "                                       data, so shaders whose resources follow from it are\n"
	         "                                       built before their dispatch is reached. Default: 1.\n"
	         "  --warmup-budget-s <n>                Start the title after n seconds of loading the shader\n"
	         "                                       store (first-used first); the rest loads in the\n"
	         "                                       background. 0 = load all first. Default: 0.\n"
	         "  --spirv-compact                     Run conservative SPIR-V cleanup after emission.\n"
	         "                                       Default: off.\n"
	         "  --mem-limit-gb <n>                   When the process's real footprint (RSS + swap, the swap\n"
	         "                                       counted at its compressed size on zram hosts) passes n GB, print the\n"
	         "                                       [mem] table and drop caches; if still over 10 s later\n"
	         "                                       stop the run with a message. 0 = off. Default: 0.\n"
	         "  --fault-storm-faults <n>             A guest page whose write watch faults n times within\n"
	         "                                       --fault-storm-frames frames stays unwatched (and is\n"
	         "                                       uploaded on every pass) for --fault-storm-hold frames.\n"
	         "                                       0 = off. Default: 0.\n"
	         "  --fault-storm-frames <n>             Window of the rule above. Default: 8.\n"
	         "  --fault-storm-hold <n>               Frames a marked page stays unwatched. Default: 60.\n"
	         "  --max-spirv-words <n>                Refuse any emitted or stored shader module larger than\n"
	         "                                       n SPIR-V words before it reaches the driver. 0 = off.\n"
	         "                                       Default: 2000000.\n"
	         "  --heap-tune <n>                      Cap the native allocator at n arenas, trim freed memory\n"
	         "                                       sooner, use mmap for large blocks, and trim after the\n"
	         "                                       store load and each build burst. 0 = off. Default: 0.\n"
	         "  --driver-cache-cap-mb <n>            Do not load or save a driver pipeline cache file larger\n"
	         "                                       than n MB (it restarts from what this run creates).\n"
	         "                                       0 = no cap. Default: 0.\n"
	         "  --materialize-memo-entries <n>       Resource materializations each shader program remembers\n"
	         "                                       (by user data, bases and the bytes it read). 1 = the\n"
	         "                                       single-entry memo; 2 or more also remembers plans that\n"
	         "                                       use the resident image heap. Default: 1.\n"
	         "  --materialize-memo-stamps <n>        Write-watch the guest pages a remembered materialization\n"
	         "                                       read, so a lookup reads none of them; a page written more\n"
	         "                                       than n times is compared by its bytes. 0 = off. Default: 0.\n"
	         "  --prefetch-backoff <n>               After four scans of submitted commands that found every\n"
	         "                                       compute shader known, scan one submission in 2, 4, ... 2^n\n"
	         "                                       until a scan finds news. 0 = scan every one. Default: 4.\n"
	         "  --frame-timer-sample <n>             The per-call timers of the [frame] line read the clock for\n"
	         "                                       one call in n. 1 = every call. Default: 16.\n"
	         "  --defer-new-targets-on-stall-ms <n>  Bring-up: once no frame was recorded for n ms, a draw\n"
	         "                                       into a target not seen redrawn is skipped while its\n"
	         "                                       shaders build instead of building them on the render\n"
	         "                                       thread (what it draws may be missing). 0 = never.\n"
	         "                                       Default: 0.\n"
	         "  --catchup-max-age-ms <n>             Bring-up: a dispatch skipped by --async-compute-wait-ms\n"
	         "                                       more than n ms ago is dropped instead of replayed (its\n"
	         "                                       descriptors may have been reused). 0 = no extra limit.\n"
	         "                                       Default: 0.\n"
	         "  --runtime-table-mark <0|1>           Bring-up: 0 leaves the candidates of a written\n"
	         "                                       runtime-indexed buffer table out of GPU ownership, so\n"
	         "                                       what such a dispatch writes is not read back. Default: 1.\n"
	         "  --enable-port <names>                Turn on ports that are off by default (comma-separated):\n"
	         "                                       callsite-lookup reuses a shader lookup by the command\n"
	         "                                       packet that made it while the registers the walk reads\n"
	         "                                       and the bytes it read are unchanged.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
	if (index + 1 >= argc) {
		return false;
	}

	index++;
	out = argv[index];
	return true;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
	auto enum_value = magic_enum::enum_cast<E>(value, magic_enum::case_insensitive);
	if (!enum_value.has_value()) {
		return false;
	}

	out = enum_value.value();
	return true;
}

static bool ParseConsoleLanguage(const std::string& value, uint32_t& out) {
	uint32_t language = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), language);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    language > Config::MAX_CONSOLE_LANGUAGE) {
		return false;
	}
	out = language;
	return true;
}

static bool ParseUint32(const std::string& value, uint32_t& out) {
	uint32_t number   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseControllerColor(const std::string& value, Config::ControllerColor& out) {
	if (value.size() != 7 || value[0] != '#') {
		return false;
	}
	uint32_t rgb = 0;
	auto [end, error] = std::from_chars(value.data() + 1, value.data() + value.size(), rgb, 16);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = {static_cast<uint8_t>(rgb >> 16), static_cast<uint8_t>(rgb >> 8),
	       static_cast<uint8_t>(rgb)};
	return true;
}

static bool ParseInt32(const std::string& value, int32_t& out) {
	int32_t number    = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseUserId(const std::string& value, int32_t& out) {
	int32_t user_id   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), user_id);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    !Config::IsConfiguredUserIdValid(user_id)) {
		return false;
	}
	out = user_id;
	return true;
}

using Config::ConfigOptions;

struct Flag {
	std::string_view name;
	bool ConfigOptions::* field;
	bool                  value = true;
};

static constexpr Flag FLAGS[] = {
    {"--rd", &ConfigOptions::renderdoc_enabled},
    {"--fullscreen", &ConfigOptions::fullscreen_enabled},
    {"--audio-3d", &ConfigOptions::audio_3d_enabled},
    {"--hdr", &ConfigOptions::hdr_enabled},
    {"--hide-cursor", &ConfigOptions::hide_cursor_enabled},
    {"--stretch", &ConfigOptions::stretch_enabled},
    {"--no-splash", &ConfigOptions::splash_enabled, false},
    {"--vr", &ConfigOptions::vr_enabled},
    {"--trinity", &ConfigOptions::pro_enabled},
    {"--amd-cpu", &ConfigOptions::amd_cpu_enabled},
    {"--guest-tsc", &ConfigOptions::guest_tsc_enabled},
    {"--network-connected", &ConfigOptions::network_connected},
    {"--emulate-psn", &ConfigOptions::psn_emulation_enabled},
    {"--vulkan-validation", &ConfigOptions::vulkan_validation_enabled},
    {"--gpu-assisted-validation", &ConfigOptions::gpu_assisted_validation_enabled},
    {"--shader-validation", &ConfigOptions::shader_validation_enabled},
    {"--keep-going", &ConfigOptions::keep_going_enabled},
    {"--shader-cache", &ConfigOptions::shader_cache_enabled},
    {"--shader-cache-keep", &ConfigOptions::shader_cache_keep_enabled},
    {"--shader-cache-retry-failed", &ConfigOptions::shader_cache_retry_failed},
    {"--async-shader-compile", &ConfigOptions::async_shader_compile_enabled},
    {"--no-async-shader-compile", &ConfigOptions::async_shader_compile_enabled, false},
    {"--tessellation", &ConfigOptions::tessellation_enabled},
    {"--occlusion-queries", &ConfigOptions::occlusion_queries_enabled},
    {"--relaxed-shader-float", &ConfigOptions::relaxed_shader_float},
    {"--native-rt", &ConfigOptions::native_rt},
    {"--lod-stats", &ConfigOptions::lod_stats_enabled},
    {"--command-buffer-dump", &ConfigOptions::command_buffer_dump_enabled},
    {"--graphics-debug-dump", &ConfigOptions::graphics_debug_dump_enabled},
    {"--profile", &ConfigOptions::profiler_enabled},
    {"--spirv-debug-printf", &ConfigOptions::spirv_debug_printf_enabled},
    {"--readback-linear-images", &ConfigOptions::readback_linear_images},
    {"--readback-tiled-images", &ConfigOptions::readback_tiled_images},
    {"--readback-storage-images", &ConfigOptions::readback_storage_images},
    {"--sync-raw-image-buffers", &ConfigOptions::sync_raw_image_buffers},
    {"--no-trophy-notifications", &ConfigOptions::trophy_enabled, false},
    {"--perf-overlay", &ConfigOptions::perf_overlay_enabled},
    {"--playgo-hack", &ConfigOptions::playgo_hack_enabled},
    {"--skip-notice-screen", &ConfigOptions::skip_notice_screen},
    {"--gpu-fault-query", &ConfigOptions::gpu_fault_query_enabled},
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
    {"--redzone", &ConfigOptions::red_zone_protection_enabled},
#endif
    {"--bda-table-validate", &ConfigOptions::bda_table_validate_enabled},
    {"--shader-key-index", &ConfigOptions::shader_key_index_enabled},
    {"--guest-fault-park", &ConfigOptions::guest_fault_park},
    {"--guest-fault-report-only", &ConfigOptions::guest_fault_report_only},
    {"--spirv-compact", &ConfigOptions::spirv_compact_enabled},
    {"--gpu-debug", &ConfigOptions::gpu_debug_enabled},
};

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
	show_help = false;

	for (int i = 1; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		std::string value;

		if (arg == "--help" || arg == "-h") {
			show_help = true;
			continue;
		}

		if (const auto* flag = std::ranges::find(FLAGS, arg, &Flag::name); flag != std::end(FLAGS)) {
			options.config.*flag->field = flag->value;
			continue;
		}

		if (Config::ParseTraceCallsOption(arg, options.config)) {
			continue;
		}

		if (!arg.starts_with("--")) {
			::printf("game input must be provided with --game\n");
			return false;
		}

		if (!NextArg(argc, argv, i, value)) {
			::printf("missing value for %s\n", arg.c_str());
			return false;
		}

		if (arg == "--game") {
			if (!options.app0_dir.empty()) {
				::printf("--game can only be specified once\n");
				return false;
			}

			value           = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (Common::File::IsDirectoryExisting(path)) {
				options.app0_dir = path;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::IsSupportedArchive(path) && Common::File::IsFileExisting(path)) {
				const auto root = Common::MakeArchivePath(path);
				if (!Common::File::IsFileExisting(root / "eboot.bin")) {
					::printf("Archive does not contain eboot.bin: %s\n", value.c_str());
					return false;
				}
				options.app0_dir = root;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::File::IsFileExisting(path)) {
				options.app0_dir = path.parent_path();

				if (options.app0_dir.empty()) {
					options.app0_dir = ".";
				}

				options.elf = std::filesystem::path("/app0") / path.filename();
			} else {
				::printf("--game must point to an existing directory, ELF, or archive: %s\n",
				         value.c_str());
				return false;
			}
		} else if (arg == "--guest-exec") {
			options.elf = Common::PathFromUtf8(Common::FixFilenameSlash(value));
			options.config.guest_arguments.assign(argv + i + 1, argv + argc);
			break;
		} else if (arg == "--game-patch") {
			if (!options.game_patch.empty()) {
				::printf("--game-patch can only be specified once\n");
				return false;
			}
			value = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (!Common::File::IsFileExisting(path)) {
				::printf("--game-patch must point to an existing file: %s\n", value.c_str());
				return false;
			}
			options.game_patch = path;
		} else if (arg == "--addcont-dir") {
			if (!options.addcont_dir.empty()) {
				::printf("--addcont-dir can only be specified once\n");
				return false;
			}
			options.addcont_dir = Common::PathFromUtf8(Common::FixFilenameSlash(value));
		} else if (arg == "--addcont") {
			auto entry = Common::ParseAddContent(value);
			if (!entry) {
				::printf("--addcont needs <16-character A-Z/0-9 label>[=<existing folder>]: %s\n",
				         value.c_str());
				return false;
			}
			options.addcont.push_back(std::move(*entry));
		} else if (arg == "--screen-width") {
			if (!ParseUint32(value, options.config.screen_width) ||
			    options.config.screen_width == 0) {
				::printf("invalid screen width: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--screen-height") {
			if (!ParseUint32(value, options.config.screen_height) ||
			    options.config.screen_height == 0) {
				::printf("invalid screen height: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--user-name") {
			if (value.empty() || value.size() > Config::MAX_USER_NAME_LENGTH) {
				::printf("invalid user name: must contain 1-%zu bytes\n",
				         Config::MAX_USER_NAME_LENGTH);
				return false;
			}
			options.config.user_name = value;
		} else if (arg == "--launch-activity") {
			if (value.empty() || value.size() > 32 ||
			    !std::ranges::all_of(value, [](char c) { return c >= ' ' && c <= '~'; })) {
				::printf("invalid activity ID: must contain 1-32 printable ASCII characters\n");
				return false;
			}
			options.config.launch_activity = value;
		} else if (arg == "--auto-cross") {
			int32_t seconds   = 0;
			auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), seconds);
			if (error != std::errc() || end != value.data() + value.size() || seconds < 0) {
				return false;
			}
			options.config.auto_cross_seconds = seconds;
		} else if (arg == "--skip-shaders") {
			std::string text;
			if (value.starts_with("@")) {
				const auto path =
				    Common::PathFromUtf8(Common::FixFilenameSlash(value.substr(1)));
				std::ifstream file(path);
				if (!file.is_open()) {
					::printf("could not open --skip-shaders file: %s\n", value.c_str() + 1);
					return false;
				}
				std::ostringstream ss;
				ss << file.rdbuf();
				text = ss.str();
			} else {
				text = value;
			}
			if (!Config::ParseShaderHashList(text, options.config.skip_shader_hashes)) {
				::printf("invalid shader hash list for --skip-shaders: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--user-id") {
			if (!ParseUserId(value, options.config.user_id)) {
				::printf("invalid user ID: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--mic") {
			options.config.audio_input_device = value;
		} else if (arg == "--audio-output") {
			if (!ParseEnum(value, options.config.audio_output)) {
				::printf("invalid audio output: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--controller-color") {
			Config::ControllerColor color {};
			if (!ParseControllerColor(value, color)) {
				::printf("invalid controller color (expected #RRGGBB): %s\n", value.c_str());
				return false;
			}
			options.config.controller_color = color;
		} else if (arg == "--controller-volume") {
			if (!ParseUint32(value, options.config.controller_speaker_volume) ||
			    options.config.controller_speaker_volume > 100) {
				::printf("invalid controller volume: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--controller-vibration") {
			if (!ParseUint32(value, options.config.controller_vibration_intensity) ||
			    options.config.controller_vibration_intensity > 100) {
				::printf("invalid controller vibration intensity: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--keyboard-player") {
			if (!ParseUint32(value, options.config.keyboard_player) ||
			    options.config.keyboard_player > static_cast<uint32_t>(Config::MAX_PLAYERS)) {
				::printf("invalid keyboard player: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--present-mode") {
			if (!ParseEnum(value, options.config.present_mode)) {
				::printf("invalid present mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--pssr-upscaler") {
			if (!ParseEnum(value, options.config.pssr_upscaler)) {
				::printf("invalid PSSR upscaler: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--dlss-folder") {
			options.config.dlss_folder = Common::PathFromUtf8(value);
		} else if (arg == "--dlss-preset") {
			if (value.size() != 1 || value[0] < 'A' || value[0] > 'O') {
				::printf("invalid DLSS preset: %s\n", value.c_str());
				return false;
			}
			options.config.dlss_preset = static_cast<uint32_t>(value[0] - 'A' + 1);
		} else if (arg == "--fsr-folder") {
			options.config.fsr_folder = Common::PathFromUtf8(value);
		} else if (arg == "--gpu") {
			if (!ParseInt32(value, options.config.gpu_index)) {
				::printf("invalid gpu index: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vblank-frequency") {
			if (!ParseUint32(value, options.config.vblank_frequency)) {
				::printf("invalid vblank frequency: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--flush-draws") {
			if (!ParseUint32(value, options.config.flush_draws)) {
				::printf("invalid draw count: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--trophy-duration") {
			if (!ParseUint32(value, options.config.trophy_duration_seconds)) {
				::printf("invalid trophy duration: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--console-language") {
			if (!ParseConsoleLanguage(value, options.config.console_language)) {
				::printf("invalid console language: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-direction") {
			if (!ParseEnum(value, options.config.shader_log_direction)) {
				::printf("invalid shader log direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-folder") {
			options.config.shader_log_folder = Common::PathFromUtf8(value);
		} else if (arg == "--command-buffer-dump-folder") {
			options.config.command_buffer_dump_folder = Common::PathFromUtf8(value);
		} else if (arg == "--printf-direction") {
			if (!ParseEnum(value, options.config.printf_direction)) {
				::printf("invalid printf direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = Common::PathFromUtf8(value);
		} else if (arg == "--keymap") {
			const auto split = value.find('=');
			if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
				::printf("invalid keymap: %s\n", value.c_str());
				return false;
			}
			options.config.keymap.push_back(value);
		} else if (arg == "--hang-dump-wait") {
			if (!ParseUint32(value, options.config.hang_dump_wait_seconds)) {
				::printf("invalid hang dump wait: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--buffer-probe") {
			if (!Config::ParseShaderHashList(value, options.config.buffer_probe_hashes)) {
				::printf("invalid shader hash list for --buffer-probe: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--loop-guard") {
			if (!ParseUint32(value, options.config.loop_guard_limit)) {
				::printf("invalid loop guard limit: %s\n", value.c_str());
				return false;
			}
			options.config.loop_guard_explicit = true;
		} else if (arg == "--disable-port") {
			if (!Config::ParseDisabledPorts(value, options.config.disabled_ports)) {
				::printf("unknown port in --disable-port: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--async-pump-budget-ms") {
			if (!ParseUint32(value, options.config.async_pump_budget_ms)) {
				::printf("invalid async pump budget: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--async-workers") {
			if (!ParseUint32(value, options.config.async_workers)) {
				::printf("invalid async worker count: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--async-speculative-modules") {
			if (!ParseUint32(value, options.config.async_speculative_modules)) {
				::printf("invalid async speculative modules value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--render-stall-report-s") {
			if (!ParseUint32(value, options.config.render_stall_report_s)) {
				::printf("invalid render stall report value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--async-ir-budget-mb") {
			if (!ParseUint32(value, options.config.async_ir_budget_mb)) {
				::printf("invalid async IR budget value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--guest-clock-hold-on-stall") {
			if (!ParseUint32(value, options.config.guest_clock_hold_ms)) {
				::printf("invalid guest clock hold value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--dump-registered-shaders") {
			if (value.empty()) {
				::printf("invalid registered shader dump directory\n");
				return false;
			}
			options.config.dump_registered_shaders_folder = Common::PathFromUtf8(value);
		} else if (arg == "--async-compute-wait-ms") {
			if (!ParseUint32(value, options.config.async_compute_wait_ms)) {
				::printf("invalid async compute wait value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--async-lookahead") {
			if (!ParseUint32(value, options.config.async_lookahead)) {
				::printf("invalid async lookahead value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--warmup-budget-s") {
			if (!ParseUint32(value, options.config.warmup_budget_s)) {
				::printf("invalid warmup budget value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--mem-limit-gb") {
			if (!ParseUint32(value, options.config.mem_limit_gb)) {
				::printf("invalid memory limit value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--fault-storm-faults") {
			if (!ParseUint32(value, options.config.fault_storm_faults)) {
				::printf("invalid fault storm faults value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--fault-storm-frames") {
			if (!ParseUint32(value, options.config.fault_storm_frames)) {
				::printf("invalid fault storm frames value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--fault-storm-hold") {
			if (!ParseUint32(value, options.config.fault_storm_hold)) {
				::printf("invalid fault storm hold value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--max-spirv-words") {
			if (!ParseUint32(value, options.config.max_spirv_words)) {
				::printf("invalid max SPIR-V words value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--heap-tune") {
			if (!ParseUint32(value, options.config.heap_tune_arenas)) {
				::printf("invalid heap tune value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--driver-cache-cap-mb") {
			if (!ParseUint32(value, options.config.driver_cache_cap_mb)) {
				::printf("invalid driver cache cap value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--materialize-memo-entries") {
			if (!ParseUint32(value, options.config.materialize_memo_entries) ||
			    options.config.materialize_memo_entries == 0u) {
				::printf("invalid materialize memo entries value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--materialize-memo-stamps") {
			if (!ParseUint32(value, options.config.materialize_memo_stamps)) {
				::printf("invalid materialize memo stamps value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--prefetch-backoff") {
			if (!ParseUint32(value, options.config.prefetch_backoff)) {
				::printf("invalid prefetch backoff value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--frame-timer-sample") {
			if (!ParseUint32(value, options.config.frame_timer_sample) ||
			    options.config.frame_timer_sample == 0u) {
				::printf("invalid frame timer sample value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--defer-new-targets-on-stall-ms") {
			if (!ParseUint32(value, options.config.defer_new_targets_on_stall_ms)) {
				::printf("invalid defer new targets on stall value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--catchup-max-age-ms") {
			if (!ParseUint32(value, options.config.catchup_max_age_ms)) {
				::printf("invalid catchup max age value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--runtime-table-mark") {
			if (!ParseUint32(value, options.config.runtime_table_mark) ||
			    options.config.runtime_table_mark > 1) {
				::printf("invalid runtime table mark value: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--enable-port") {
			if (!Config::ParseDisabledPorts(value, options.config.enabled_ports)) {
				::printf("unknown port in --enable-port: %s\n", value.c_str());
				return false;
			}
		} else {
			::printf("unknown option: %s\n", arg.c_str());
			return false;
		}
	}

	if (options.config.gpu_assisted_validation_enabled) {
		options.config.vulkan_validation_enabled = true;
	}

	return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

static int Main(int argc, char* argv[]) {
	InitializeThreads();

	RunOptions options;
	bool       show_help = false;

	if (argc < 2) {
		PrintUsage();
		return 0;
	}

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		return 1;
	}

	if (show_help) {
		PrintUsage();
		return 0;
	}

	Run(options);

	return 0;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

int wmain(int argc, wchar_t* argv[]) {
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<size_t>(argc));

    for (int index = 0; index < argc; index++) {
        const std::wstring_view wide(argv[index]);
        const std::u16string utf16(wide.begin(), wide.end());

        utf8_args.push_back(Common::Utf16ToUtf8(utf16));
    }

    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_args.size());

    for (auto& argument: utf8_args) {
        utf8_argv.push_back(argument.data());
    }

    return Main(argc, utf8_argv.data());
}

#else

int main(int argc, char* argv[]) {
    return Main(argc, argv);
}

#endif
