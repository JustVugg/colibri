import { afterEach, describe, expect, it, vi } from "vitest"

import { FILE_TOOLS, approvalText, runFileTool } from "./files"

afterEach(() => vi.unstubAllGlobals())

const options = (approve = () => true) => ({
  baseUrl: "http://127.0.0.1:8000/v1", apiKey: "", folder: "/tmp/w", signal: new AbortController().signal, approve,
})
const call = (name: string, args: Record<string, unknown>) => ({ id: "c", name, arguments: JSON.stringify(args) })

describe("file tools", () => {
  it("offers five tools, each with its required arguments", () => {
    expect(FILE_TOOLS.map((tool) => [tool.function.name, tool.function.parameters.required])).toEqual([
      ["read_file", ["path"]], ["write_file", ["path", "content"]], ["edit_file", ["path", "old_text", "new_text"]],
      ["list_dir", ["path"]], ["run_command", ["command"]],
    ])
  })

  it("asks before a write, an edit and a command, never before a read or a list", () => {
    expect(approvalText(call("run_command", { command: "make" }))).toBe("Run this command?\n\nmake")
    expect(approvalText(call("write_file", { path: "a.py", content: "abc" }))).toBe("Write this file?\n\na.py\n3 characters")
    expect(approvalText(call("edit_file", { path: "a.py", old_text: "x = 1", new_text: "x = 2" })))
      .toBe("Edit this file?\n\na.py\n\nReplace:\nx = 1\n\nWith:\nx = 2")
    expect(approvalText(call("read_file", { path: "a.py" }))).toBeNull()
    expect(approvalText(call("list_dir", { path: "." }))).toBeNull()
  })

  it("runs a call on the gateway's /fs route with the working folder", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ cwd: "/tmp/w", exit: 0, output: "ok\n" })))
    vi.stubGlobal("fetch", fetchMock)
    const result = await runFileTool(call("run_command", { command: "make" }), options())
    expect(fetchMock.mock.calls[0][0]).toBe("http://127.0.0.1:8000/fs/run")
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ command: "make", folder: "/tmp/w" })
    expect(result).toEqual({ content: "exit 0 in /tmp/w\nok\n", label: "run_command make · exit 0" })
  })

  it("an edit goes to /fs/edit and reports the lines it changed", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ path: "/tmp/w/a.py", removed: 1, added: 2 })))
    vi.stubGlobal("fetch", fetchMock)
    const result = await runFileTool(call("edit_file", { path: "a.py", old_text: "x", new_text: "y\nz" }), options())
    expect(fetchMock.mock.calls[0][0]).toBe("http://127.0.0.1:8000/fs/edit")
    expect(result).toEqual({ content: "Edited /tmp/w/a.py: 1 lines replaced by 2", label: "edit_file a.py · −1 +2" })
  })

  it("a declined call never reaches the gateway", async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    const result = await runFileTool(call("write_file", { path: "a.py", content: "x" }), options(() => false))
    expect(fetchMock).not.toHaveBeenCalled()
    expect(result.content).toBe("Error: declined by the user.")
  })

  it("a gateway error goes back to the model as text", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ error: { message: "FileNotFoundError: nope.py" } }), { status: 400 })))
    const result = await runFileTool(call("read_file", { path: "nope.py" }), options())
    expect(result.content).toBe("Error: FileNotFoundError: nope.py")
  })

  it("an unknown tool is an error, not a request", async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    expect((await runFileTool(call("delete_all", {}), options())).content).toBe("Error: there is no tool named delete_all.")
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
