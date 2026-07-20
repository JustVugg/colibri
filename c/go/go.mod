// A dependency-free Go port of the colibrì `coli` launcher (issue #310).
//
// Deliberately zero third-party requires: Colibri's runtime path is
// dependency-free by design (CONTRIBUTING.md), and the Python gateway is
// stdlib-only for the same reason. This module uses only the Go standard
// library so the resulting `coli` binary is a single static executable.
module github.com/JustVugg/colibri/coli

go 1.21
