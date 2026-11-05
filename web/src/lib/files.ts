import { serverEndpoint, type ToolCall, type ToolSpec } from "./api"

/* The file tools the chat offers the model when the gateway runs them
   (COLI_FILE_TOOLS=1, /health file_tools). The model calls them through the
   gateway's OpenAI tool_calls, whatever its native format; the page runs each
   call on POST /fs/<action>, asks the user before every write and command, and
   sends the result back as a role "tool" turn. */

export const FILE_TOOL_ROUNDS = 25

const ACTIONS: Record<string, "read" | "write" | "edit" | "list" | "cd" | "run"> = {
  read_file: "read", write_file: "write", edit_file: "edit", list_dir: "list", change_dir: "cd", run_command: "run",
}

const fn = (name: string, description: string, properties: Record<string, { type: string; description: string }>) =>
  ({ type: "function" as const, function: { name, description, parameters: { type: "object", properties, required: Object.keys(properties) } } })

export const FILE_TOOLS: ToolSpec[] = [
  fn("read_file", "Read a text file. A relative path is inside the working folder.",
    { path: { type: "string", description: "the file to read" } }),
  fn("write_file", "Create or overwrite a text file with its whole new content. Missing folders are created.",
    { path: { type: "string", description: "the file to write" }, content: { type: "string", description: "the complete new content of the file" } }),
  fn("edit_file", "Change part of a text file: replace old_text, which must occur exactly once in the file, with new_text. Read the file first and copy old_text exactly, with enough lines to be unique.",
    { path: { type: "string", description: "the file to change" }, old_text: { type: "string", description: "the exact text to replace, as it is in the file" }, new_text: { type: "string", description: "the text to put in its place" } }),
  fn("list_dir", "List a folder: one name a line, folders end with /.",
    { path: { type: "string", description: "the folder, . for the working folder" } }),
  fn("change_dir", "Change the working folder, like cd: every later tool runs there until it changes again. Use it whenever the user asks to cd or to go to a folder.",
    { path: { type: "string", description: "the folder to go to, absolute or relative to the working folder" } }),
  fn("run_command", "Run a shell command in the working folder: its exit code and output (stdout and stderr), at most 120 s, no input.",
    { command: { type: "string", description: "the shell command" } }),
]

/* The same text in every request: it sits at the top of the prompt, so naming the working
   folder here would change the prompt after each change_dir and make the server read the
   whole conversation again. The folder is in each tool's result instead. */
export function fileToolsSystem(): string {
  return "You can work on the user's files, in a working folder: read_file, write_file, edit_file and list_dir " +
    "take paths relative to it, and run_command runs a shell command there. Each tool's result names the folder " +
    "it used; list_dir . shows where you are. " +
    "A cd inside run_command does not last; to change the working folder call change_dir. " +
    "Never say you changed folder, listed, read or ran something unless the tool did it in this turn. " +
    "Read a file before you change it. To change part of a file use edit_file; " +
    "use write_file only for a new file or to replace a whole file. " +
    "Only write the content the user asked for: a file asked for without content is created empty. " +
    "The user approves each write and each command; " +
    "if one is declined, do not try it again: say what you wanted to do."
}

export interface FileToolResult {
  /* sent back to the model */
  content: string
  /* the line the page shows: "write_file src/a.py", "run_command make · exit 0" */
  label: string
  /* change_dir: the new working folder, absolute, as the gateway checked it */
  folder?: string
}

function parseArgs(call: ToolCall): Record<string, unknown> {
  try {
    const value = JSON.parse(call.arguments || "{}")
    return value && typeof value === "object" ? value as Record<string, unknown> : {}
  } catch {
    return {}
  }
}

/* What the user is asked before a write or a command; null for a read or a list. */
export function approvalText(call: ToolCall): string | null {
  const args = parseArgs(call)
  if (ACTIONS[call.name] === "run") return `Run this command?\n\n${String(args.command ?? "")}`
  if (ACTIONS[call.name] === "write") return `Write this file?\n\n${String(args.path ?? "")}\n${String(args.content ?? "").length} characters`
  if (ACTIONS[call.name] === "edit") {
    const cut = (text: unknown) => { const s = String(text ?? ""); return s.length > 600 ? `${s.slice(0, 600)}\n…` : s }
    return `Edit this file?\n\n${String(args.path ?? "")}\n\nReplace:\n${cut(args.old_text)}\n\nWith:\n${cut(args.new_text)}`
  }
  return null
}

export interface RunFileToolOptions {
  baseUrl: string
  apiKey: string
  folder: string
  signal: AbortSignal
  /* asked before a write or a command; false declines the call */
  approve: (question: string) => boolean
}

export async function runFileTool(call: ToolCall, options: RunFileToolOptions): Promise<FileToolResult> {
  const action = ACTIONS[call.name]
  const args = parseArgs(call)
  const subject = String(args.command ?? args.path ?? ".")
  const label = `${call.name} ${subject}`
  if (!action) return { content: `Error: there is no tool named ${call.name}.`, label }
  const question = approvalText(call)
  if (question && !options.approve(question)) return { content: "Error: declined by the user.", label: `${label} · declined` }
  try {
    const response = await fetch(serverEndpoint(options.baseUrl, `fs/${action}`), {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}) },
      body: JSON.stringify({ ...args, folder: options.folder }),
      signal: options.signal,
    })
    const body = await response.json() as Record<string, unknown> & { error?: { message?: string } }
    if (!response.ok) throw new Error(body.error?.message || `${response.status} ${response.statusText}`)
    if (action === "read") return { content: `${body.path}\n\n${body.text}`, label }
    if (action === "write") return { content: `Wrote ${body.bytes} bytes to ${body.path}`, label }
    if (action === "edit") return { content: `Edited ${body.path}: ${body.removed} lines replaced by ${body.added}`, label: `${label} · −${body.removed} +${body.added}` }
    if (action === "list") {
      const entries = (body.entries as string[]) || []
      return { content: `${body.path}\n${entries.join("\n")}${body.more ? `\n[${body.more} more]` : ""}`, label: `${label} · ${body.path}` }
    }
    if (action === "cd") return { content: `The working folder is now ${body.path}`, label: `${label} · ${body.path}`, folder: String(body.path) }
    const exit = body.exit ?? "timeout"
    return { content: `exit ${exit} in ${body.cwd}\n${body.output}`, label: `${label} · exit ${exit}` }
  } catch (cause) {
    if (options.signal.aborted) throw cause
    const message = cause instanceof Error ? cause.message : String(cause)
    return { content: `Error: ${message}`, label: `${label} · ${message}` }
  }
}
