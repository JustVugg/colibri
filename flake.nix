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

        colibri = pkgs.callPackage ./package.nix {};
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

          packages = with pkgs;
            [
              colibri.pythonEnv
              gcc
              gnumake
              clang-tools # clangd / clang-tidy for IDE support
              pkg-config
            ]
            ++ lib.optionals stdenv.hostPlatform.isLinux [
              vulkan-loader
              vulkan-headers
              vulkan-tools
              shaderc
            ];

          shellHook =
            pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.vulkan-loader]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ''
            + ''
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
