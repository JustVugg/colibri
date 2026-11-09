import sys
import os
import zipfile
import json
import toml
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python jar_to_hnmod.py INPUT.jar [-o OUTPUT.hnmod]")
        return 1

    input_path = sys.argv[1]
    output_path = None

    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == "-o":
            output_path = sys.argv[i + 1]
            break

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".hnmod"

    try:
        convert_jar_to_hnmod(input_path, output_path)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

def convert_jar_to_hnmod(input_path, output_path):
    with zipfile.ZipFile(input_path, 'r') as jar:
        # Extract metadata
        metadata = extract_metadata(jar)
        
        # Initialize counts
        counts = {"assets": 0, "data": 0, "classes": 0, "resources": 0}
        
        # Create output zip
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as hnmod:
            # Write manifest
            manifest = {
                "format": "hnmod",
                "format_version": 1,
                "id": metadata["id"],
                "name": metadata["name"],
                "version": metadata["version"],
                "description": metadata["description"],
                "authors": metadata["authors"],
                "license": metadata["license"],
                "source_loader": metadata["source_loader"],
                "source_file": os.path.basename(input_path),
                "dependencies": metadata["dependencies"],
                "counts": counts
            }
            hnmod.writestr("hnmod.json", json.dumps(manifest, indent=2))
            
            # Process files
            for info in jar.infolist():
                if info.is_dir():
                    continue
                
                name = info.filename
                data = jar.read(info)
                
                if name.startswith("assets/"):
                    hnmod.writestr(f"assets/{name[7:]}", data)
                    counts["assets"] += 1
                elif name.startswith("data/"):
                    hnmod.writestr(f"data/{name[5:]}", data)
                    counts["data"] += 1
                elif name.endswith(".class"):
                    hnmod.writestr(f"code/{name}", data)
                    counts["classes"] += 1
                else:
                    hnmod.writestr(f"resources/{name}", data)
                    counts["resources"] += 1

def extract_metadata(jar):
    # Try to find metadata files
    metadata_files = {
        "META-INF/neoforge.mods.toml": "neoforge",
        "META-INF/mods.toml": "forge",
        "fabric.mod.json": "fabric",
        "quilt.mod.json": "quilt",
        "mcmod.info": "forge"
    }
    
    for path, loader in metadata_files.items():
        try:
            data = jar.read(path)
            if loader == "neoforge":
                return parse_neoforge_metadata(data)
            elif loader == "forge":
                return parse_forge_metadata(data)
            elif loader == "fabric":
                return parse_fabric_metadata(data)
            elif loader == "quilt":
                return parse_quilt_metadata(data)
            elif loader == "forge":
                return parse_forge_metadata(data)
        except KeyError:
            continue
    
    # No metadata found
    base = os.path.splitext(os.path.basename(jar.filename))[0]
    return {
        "id": base,
        "name": base,
        "version": "0.0.0",
        "description": "",
        "authors": [],
        "license": "",
        "source_loader": "unknown",
        "dependencies": []
    }

def parse_neoforge_metadata(data):
    try:
        toml_data = toml.loads(data.decode('utf-8'))
        mod_table = toml_data.get("mods", [{}])[0]
        mod_id = mod_table.get("modId", "")
        version = mod_table.get("version", "")
        name = mod_table.get("displayName", "")
        description = mod_table.get("description", "")
        authors = mod_table.get("authors", "")
        license = mod_table.get("license", "")
        
        # Parse authors
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",")]
        elif isinstance(authors, list):
            authors = [a.strip() for a in authors]
        else:
            authors = []
        
        # Parse dependencies
        dependencies = []
        deps_table = mod_table.get("dependencies", {})
        for mod_id, dep_info in deps_table.items():
            if isinstance(dep_info, dict):
                dep = {
                    "id": mod_id,
                    "type": dep_info.get("type", "required"),
                    "version_range": dep_info.get("versionRange", "*")
                }
                dependencies.append(dep)
        
        return {
            "id": mod_id,
            "name": name,
            "version": version,
            "description": description,
            "authors": authors,
            "license": license,
            "source_loader": "neoforge",
            "dependencies": dependencies
        }
    except Exception:
        return {
            "id": "unknown",
            "name": "unknown",
            "version": "0.0.0",
            "description": "",
            "authors": [],
            "license": "",
            "source_loader": "neoforge",
            "dependencies": []
        }

def parse_forge_metadata(data):
    try:
        toml_data = toml.loads(data.decode('utf-8'))
        mod_table = toml_data.get("mods", [{}])[0]
        mod_id = mod_table.get("modId", "")
        version = mod_table.get("version", "")
        name = mod_table.get("displayName", "")
        description = mod_table.get("description", "")
        authors = mod_table.get("authors", "")
        license = mod_table.get("license", "")
        
        # Parse authors
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",")]
        elif isinstance(authors, list):
            authors = [a.strip() for a in authors]
        else:
            authors = []
        
        # Parse dependencies
        dependencies = []
        deps_table = mod_table.get("dependencies", {})
        for mod_id, dep_info in deps_table.items():
            if isinstance(dep_info, dict):
                dep = {
                    "id": mod_id,
                    "type": dep_info.get("type", "required"),
                    "version_range": dep_info.get("versionRange", "*")
                }
                dependencies.append(dep)
        
        return {
            "id": mod_id,
            "name": name,
            "version": version,
            "description": description,
            "authors": authors,
            "license": license,
            "source_loader": "forge",
            "dependencies": dependencies
        }
    except Exception:
        return {
            "id": "unknown",
            "name": "unknown",
            "version": "0.0.0",
            "description": "",
            "authors": [],
            "license": "",
            "source_loader": "forge",
            "dependencies": []
        }

def parse_fabric_metadata(data):
    try:
        json_data = json.loads(data.decode('utf-8'))
        mod_id = json_data.get("id", "")
        name = json_data.get("name", "")
        version = json_data.get("version", "")
        description = json_data.get("description", "")
        authors = json_data.get("authors", [])
        license = json_data.get("license", "")
        
        # Flatten authors
        if isinstance(authors, list):
            flat_authors = []
            for a in authors:
                if isinstance(a, str):
                    flat_authors.append(a)
                elif isinstance(a, dict) and "name" in a:
                    flat_authors.append(a["name"])
            authors = flat_authors
        else:
            authors = []
        
        # Parse dependencies
        dependencies = []
        depends = json_data.get("depends", {})
        for mod_id, version_range in depends.items():
            dep = {
                "id": mod_id,
                "type": "required",
                "version_range": str(version_range)
            }
            dependencies.append(dep)
        
        return {
            "id": mod_id,
            "name": name,
            "version": version,
            "description": description,
            "authors": authors,
            "license": license,
            "source_loader": "fabric",
            "dependencies": dependencies
        }
    except Exception:
        return {
            "id": "unknown",
            "name": "unknown",
            "version": "0.0.0",
            "description": "",
            "authors": [],
            "license": "",
            "source_loader": "fabric",
            "dependencies": []
        }

def parse_quilt_metadata(data):
    try:
        json_data = json.loads(data.decode('utf-8'))
        loader_data = json_data.get("quilt_loader", {})
        mod_id = loader_data.get("id", "")
        version = loader_data.get("version", "")
        
        metadata = json_data.get("metadata", {})
        name = metadata.get("name", "")
        description = metadata.get("description", "")
        authors = metadata.get("authors", [])
        license = metadata.get("license", "")
        
        # Flatten authors
        if isinstance(authors, list):
            flat_authors = []
            for a in authors:
                if isinstance(a, str):
                    flat_authors.append(a)
                elif isinstance(a, dict) and "name" in a:
                    flat_authors.append(a["name"])
            authors = flat_authors
        else:
            authors = []
        
        return {
            "id": mod_id,
            "name": name,
            "version": version,
            "description": description,
            "authors": authors,
            "license": license,
            "source_loader": "quilt",
            "dependencies": []
        }
    except Exception:
        return {
            "id": "unknown",
            "name": "unknown",
            "version": "0.0.0",
            "description": "",
            "authors": [],
            "license": "",
            "source_loader": "quilt",
            "dependencies": []
        }

if __name__ == "__main__":
    sys.exit(main())
