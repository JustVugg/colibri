import type { ChatMessage } from "./api"

/* Stream a delta into one bubble of the transcript. Both a fresh answer and a
   continued turn go through here; the only difference is which bubble. */
export function appendDelta(messages: ChatMessage[], targetId: string, field: "content" | "reasoning", delta: string): ChatMessage[] {
  return messages.map((item) =>
    item.id === targetId ? { ...item, [field]: (item[field] ?? "") + delta } : item)
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
