{
  description = "colibrì — run large MoE models (GLM-5.2, OLMoE, DeepSeek V4 Flash) on a consumer machine";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    inputs:
    let
      inherit (inputs.nixpkgs) lib;
    in
    {
      packages = builtins.mapAttrs (system: pkgs: {
        colibri = pkgs.callPackage ./package.nix { };
        default = inputs.self.packages.${system}.colibri;
      }) inputs.nixpkgs.legacyPackages;

      apps = builtins.mapAttrs (system: _: {
        # `nix run .#engine` runs the engine binary directly, skipping the
        # coli launcher. Named "engine", not "colibri", so it doesn't shadow
        # packages.colibri (whose mainProgram is coli).
        engine = {
          type = "app";
          program = "${inputs.self.packages.${system}.colibri}/bin/colibri";
        };
      }) inputs.nixpkgs.legacyPackages;

      formatter = builtins.mapAttrs (
        _: pkgs:
        pkgs.treefmt.withConfig {
          settings = {
            tree-root-file = "flake.nix";
            on-unmatched = "info";
            formatter = {
              nixfmt = {
                command = lib.getExe pkgs.nixfmt;
                includes = [ "*.nix" ];
              };
              statix = {
                command = lib.getExe pkgs.statix;
                options = [ "fix" ];
                no-positional-arg-support = true;
                includes = [ "*.nix" ];
              };
              deadnix = {
                command = lib.getExe pkgs.deadnix;
                options = [ "--edit" ];
                includes = [ "*.nix" ];
              };
              rustfmt = {
                command = lib.getExe pkgs.rustfmt;
                options = [
                  "--config"
                  "skip_children=true"
                  "--edition"
                  "2024"
                ];
                includes = [ "*.rs" ];
              };
              # Keeps .editorconfig's four spaces and the hand-laid-out lists.
              taplo = {
                command = lib.getExe pkgs.taplo;
                options = [
                  "format"
                  "--option"
                  "indent_string=    "
                  "--option"
                  "array_auto_collapse=false"
                ];
                includes = [ "*.toml" ];
              };
            };
          };
        }
      ) inputs.nixpkgs.legacyPackages;

      checks = builtins.mapAttrs (system: _: {
        formatting = inputs.self.formatter.${system}.check inputs.self;
      }) inputs.nixpkgs.legacyPackages;

      devShells = builtins.mapAttrs (
        system: pkgs:
        let
          inherit (inputs.self.packages.${system}) colibri;
        in
        {
          default = pkgs.mkShell {
            inputsFrom = [ colibri ];

            packages =
              with pkgs;
              [
                colibri.pythonEnv
                inputs.self.formatter.${system}
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
                export LD_LIBRARY_PATH="${
                  pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader ]
                }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
      ) inputs.nixpkgs.legacyPackages;
    };
}
