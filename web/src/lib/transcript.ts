import type { ChatMessage } from "./api"

/* Stream a delta into one bubble of the transcript. Both a fresh answer and a
   continued turn go through here; the only difference is which bubble. */
export function appendDelta(messages: ChatMessage[], targetId: string, field: "content" | "reasoning", delta: string): ChatMessage[] {
  return messages.map((item) =>
    item.id === targetId ? { ...item, [field]: (item[field] ?? "") + delta } : item)
}

/* Record how the generation streaming into `targetId` ended. A null finish is
   a stream that closed without a finish_reason: colibri always sends one last,
   so the reply was cut off, and is recorded as "incomplete", not as done. */
export function setFinish(messages: ChatMessage[], targetId: string, finish: string | null): ChatMessage[] {
  return messages.map((item) => item.id === targetId ? { ...item, finish: finish ?? "incomplete" } : item)
}

/* An assistant turn is open, and worth continuing, when it hit max_tokens or
   was cut off by a stop, an error, or a stream that closed without saying how
   it finished. "stop" means the model closed the turn,
   and continuing it only makes the model re-emit its stop and add nothing. */
export function continuable(message: ChatMessage): boolean {
  return message.role === "assistant" && !!message.content.trim()
    && ["length", "aborted", "error", "incomplete"].includes(message.finish ?? "")
}

/* The request that continues the trailing assistant turn instead of opening a
   new one: the transcript ending on that turn, streamed back into the same
   bubble. Trailing whitespace is stripped because the server refuses it (the
   template strips it, so the model would resume from different bytes than were
   sent); the returned history carries those exact bytes, so the bubble shows
   what it resumes from. null when there is no non-empty assistant turn last. */
export function continuation(messages: ChatMessage[]): { history: ChatMessage[]; targetId: string } | null {
  const last = messages[messages.length - 1]
  if (!last || last.role !== "assistant" || !last.content.trim()) return null
  const trimmed = last.content.replace(/\s+$/, "")
  return {
    history: messages.map((item) => item.id === last.id ? { ...item, content: trimmed } : item),
    targetId: last.id,
  }
}
