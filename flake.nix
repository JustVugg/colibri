{
  description = "colibrì — run GLM-5.2 (744B MoE) on a consumer machine with ~25 GB RAM";

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
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfreePredicate = pkg:
            nixpkgs.lib.any (
              license: (license.shortName or "") == "CUDA EULA"
            ) (nixpkgs.lib.toList pkg.meta.license);
        };
        cudaToolkit = pkgs.cudaPackages_13_2.cudatoolkit;

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

        colibri = {cudaSupport ? false}:
          pkgs.stdenv.mkDerivation {
            pname = "colibri";
            version = "1.0";
            src = ./.;

            nativeBuildInputs = with pkgs; [makeWrapper python3];

            buildInputs = with pkgs;
              [
                gcc
                gmp
              ]
              ++ pkgs.lib.optionals cudaSupport [cudaToolkit];

            checkInputs = [pythonEnv];

            # Use x86-64-v3 (AVX2) for a portable binary; override with ARCH=native for local builds
            ARCH =
              if pkgs.stdenv.hostPlatform.isx86_64
              then "x86-64-v3"
              else "native";

            buildPhase = ''
              runHook preBuild
              make -C c glm ARCH="$ARCH" ${pkgs.lib.optionalString cudaSupport "CUDA=1 CUDA_HOME=${cudaToolkit}"}
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall

              # Keep the CLI's Python modules beside its bundled engine.
              mkdir -p $out/lib/colibri/tools $out/bin
              cp c/glm $out/lib/colibri/glm
              cp c/coli $out/lib/colibri/coli
              chmod +x $out/lib/colibri/coli
              cp c/openai_server.py c/resource_plan.py c/doctor.py c/version.py \
                $out/lib/colibri/
              cp -r c/tools/* $out/lib/colibri/tools/
              ln -s ../lib/colibri/glm $out/bin/glm

              makeWrapper ${pythonEnv}/bin/python $out/bin/coli \
                --add-flags "$out/lib/colibri/coli" \
                --set-default COLI_ENGINE "$out/lib/colibri/glm" \
                --set PYTHONPATH "$out/lib/colibri:${pythonEnv}/${pkgs.python3.sitePackages}"
              runHook postInstall
            '';

            checkPhase = ''
              runHook preCheck
              cd c
              make test-c
              cd ..
              runHook postCheck
            '';

            doCheck = true;

            meta = with pkgs.lib; {
              description = "Run GLM-5.2 (744B MoE) on a consumer machine with ~25 GB RAM";
              homepage = "https://github.com/JustVugg/colibri";
              license = licenses.asl20;
              platforms = with platforms; linux ++ darwin;
              mainProgram = "coli";
            };
          };
      in {
        packages = {
          default = colibri {};
          colibri = colibri {};
          cuda = colibri {cudaSupport = true;};
        };

        apps = {
          default = {
            type = "app";
            program = pkgs.lib.getExe (colibri {});
          };
          glm = {
            type = "app";
            program = "${colibri {}}/lib/colibri/glm";
          };
        };

        formatter = (import nixpkgs {inherit system;}).alejandra;

        devShells.default = pkgs.mkShell {
          inputsFrom = [colibri {}];

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
            echo "Build the engine:   make -C c glm"
            echo "Run the converter:  python c/coli convert --model /path/to/glm52_i4"
            echo "Chat:               COLI_MODEL=/path/to/glm52_i4 ./c/glm ..."
          '';
        };
      }
    );
}
