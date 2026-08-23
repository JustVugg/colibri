{
  description = "colibrì — run large MoE models (GLM-5.2, OLMoE, DeepSeek V4 Flash) on a consumer machine";

  # flake.lock (committed) pins these branch inputs to exact commit SHAs,
  # so builds are reproducible; refresh with `nix flake update`.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};

        # Python with the packages needed by the offline converter tools
        pythonEnv = pkgs.python3.withPackages (
          ps:
            with ps; [
              torch
              safetensors
              huggingface-hub
              numpy
              tokenizers
              datasets
            ]
        );

        isDarwin = pkgs.stdenv.hostPlatform.isDarwin;

        # Real version from the engine's single source of truth (c/version.py).
        colibriVersion = let
          m = builtins.match ''.*__version__ = "([^"]+)".*'' (builtins.readFile ./c/version.py);
        in
          if m == null
          then "0"
          else builtins.head m;

        # Apple clang has no OpenMP runtime and the Makefile only finds libomp via
        # `brew`, absent here; expose omp.h + libomp as OMPDIR or macOS goes single-threaded.
        colibriOmp = pkgs.symlinkJoin {
          name = "colibri-openmp";
          paths = [pkgs.llvmPackages.openmp pkgs.llvmPackages.openmp.dev];
        };

        # Portable default per arch — never -mcpu=native, which would pin a
        # distributed binary to the builder's core and break substitution.
        archBaseline =
          if pkgs.stdenv.hostPlatform.isx86_64
          then "x86-64-v3"
          else if isDarwin
          then "" # arm64 macOS: NEON is baseline
          else "armv8-a";

        # Build args: portable ARCH, plus (on macOS) OpenMP via OMPDIR and the
        # Metal backend — Apple clang provides neither on its own.
        buildArgs =
          "ARCH=${archBaseline}"
          + pkgs.lib.optionalString isDarwin " OMPDIR=${colibriOmp} METAL=1";

        colibri = pkgs.stdenv.mkDerivation {
          pname = "colibri";
          version = colibriVersion;
          src = ./.;

          nativeBuildInputs = with pkgs; [makeWrapper];

          # Compiler comes from stdenv (clang on Darwin, gcc on Linux); these add
          # only the extra build/runtime libs each platform needs.
          buildInputs = with pkgs;
            lib.optionals stdenv.hostPlatform.isDarwin [
              llvmPackages.openmp # libomp runtime for the OpenMP build
              apple-sdk_15 # SDK 15 headers enable the Metal residency-set path
            ]
            ++ lib.optionals stdenv.hostPlatform.isLinux [
              stdenv.cc.cc.lib # libgomp.so.1 in the runtime closure
            ];

          buildPhase = ''
            runHook preBuild
            # `make install` builds and stages every engine it produces —
            # colibri (GLM), olmoe, and deepseek_v4 where COLI_V4_SUPPORTED —
            # beside coli under $out/lib/colibri so coli's HERE-relative
            # engine_for() finds each. inkling/kimi_k3 aren't in `install` yet.
            make -C c install ${buildArgs} \
              DESTDIR=$out PREFIX= BINDIR=/bin LIBEXECDIR=/lib/colibri
            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall

            # `make install` handled the engines, coli and the support
            # modules; two fixups remain:

            # 1. `make install`'s file lists miss two things packaged Python
            #    needs: v4_dsml.py (openai_server.py imports it unconditionally,
            #    so `coli serve` / `coli web` would ModuleNotFoundError) and
            #    tools/iq3xxs_grid.json (iq3_pack.py loads it for `coli convert
            #    --xbits e8`, with no fallback). Back them in — each guard
            #    defers to a future `make install` that stages them itself.
            [ -e "$out/lib/colibri/v4_dsml.py" ] || \
              install -m 644 c/v4_dsml.py "$out/lib/colibri/"
            [ -e "$out/lib/colibri/tools/iq3xxs_grid.json" ] || \
              install -m 644 c/tools/iq3xxs_grid.json "$out/lib/colibri/tools/"

            # 2. coli dispatches relative to its own dir, so it must sit beside
            #    the engines; that also frees $out/bin/coli for the wrapper. The
            #    engine is re-exposed as $out/bin/colibri for `nix run .#engine`.
            mv $out/bin/coli $out/lib/colibri/coli
            ln -s ../lib/colibri/colibri $out/bin/colibri

            # Wrap coli through pythonEnv. COLI_ENGINE is deliberately NOT set:
            # c/coli's engine_for() routes EVERY model to GLM whenever
            # COLI_ENGINE is present, defeating per-model dispatch. Left unset,
            # coli resolves each engine beside itself (colibri / olmoe /
            # deepseek_v4 under $out/lib/colibri). PYTHONPATH makes `import
            # openai_server` / `resource_plan` / `doctor` / `v4_dsml` resolve.
            makeWrapper ${pythonEnv}/bin/python $out/bin/coli \
              --add-flags "$out/lib/colibri/coli" \
              --set PYTHONPATH "$out/lib/colibri:${pythonEnv}/${pkgs.python3.sitePackages}"
            runHook postInstall
          '';

          # `make test-c` isn't hermetic in a sandbox (test_ssd_probe timing,
          # Linux test_uring/io_uring); installCheckPhase validates instead.
          doCheck = false;

          # Offline verification that the multi-engine layout is correct: the
          # engines reached $out, the backfilled convert data asset is present,
          # the wrapper starts (`coli --version` argparse-exits before any model
          # load), and the serve import surface (openai_server -> v4_dsml)
          # resolves. Not versionCheckHook: coli's version string is
          # unrelated to this derivation's `version`.
          doInstallCheck = true;
          installCheckPhase = ''
            runHook preInstallCheck
            # An install check must not mutate $out; block the .pyc these imports write.
            export PYTHONDONTWRITEBYTECODE=1
            test -x $out/lib/colibri/colibri
            test -x $out/lib/colibri/olmoe
            test -f $out/lib/colibri/tools/iq3xxs_grid.json
            $out/bin/coli --version
            PYTHONPATH=$out/lib/colibri ${pythonEnv}/bin/python -c 'import openai_server'
            runHook postInstallCheck
          '';

          meta = with pkgs.lib; {
            description = "Run large MoE models (GLM-5.2, OLMoE, DeepSeek V4 Flash) in pure C, experts streamed from disk";
            homepage = "https://github.com/JustVugg/colibri";
            license = licenses.asl20;
            platforms = with platforms; linux ++ darwin;
            mainProgram = "coli";
          };
        };
      in {
        packages = {
          default = colibri;
          inherit colibri;
        };

        apps = {
          default = {
            type = "app";
            program = pkgs.lib.getExe colibri;
          };
          # `nix run .#engine` runs the engine binary directly, skipping the
          # coli launcher. Named "engine", not "colibri", so it doesn't shadow
          # packages.colibri (whose mainProgram is coli).
          engine = {
            type = "app";
            program = "${colibri}/bin/colibri";
          };
        };

        formatter = pkgs.alejandra;

        devShells.default = pkgs.mkShell {
          inputsFrom = [colibri];

          packages = with pkgs; [
            pythonEnv
            gcc
            gnumake
            clang-tools # clangd / clang-tidy for IDE support
            pkg-config
          ];

          shellHook = ''
            echo "🐦 colibrì dev shell"
            echo "  gcc: $(gcc --version | head -1)"
            echo "  python: $(python3 --version)"
            echo ""
            echo "Build the engine:   make -C c colibri"
            echo "Run the converter:  python c/coli convert --model /path/to/glm52_i4"
            echo "Chat:               COLI_MODEL=/path/to/glm52_i4 ./c/colibri ..."
          '';
        };
      }
    );
}
