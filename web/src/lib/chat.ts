import type { ChatMessage } from "./api"

export interface ResendTurn {
  text: string
  previous: ChatMessage[]
  pictures: string[]
}

/* Rebuild the last user turn so regenerate can send it again. The pictures
   live on that message, not in the composer: a follow-up without them would
   get an answer about nothing, and an image-only turn would not send at all. */
export function resendFrom(messages: ChatMessage[], assistantIndex: number): ResendTurn | null {
  let userIndex = -1
  for (let i = 0; i < assistantIndex; i++) {
    if (messages[i].role === "user") userIndex = i
  }
  if (userIndex < 0) return null
  const user = messages[userIndex]
  return {
    text: user.content,
    previous: messages.slice(0, userIndex),
    pictures: user.images ? [...user.images] : [],
  }
}

export interface ChatLabels {
  user: string
  assistant: string
  reasoning: string
  image: string
}

/* The conversation as a Markdown file someone can read, keep or paste elsewhere: one
   heading a turn, the reasoning folded away as on screen, a picture as a placeholder
   (its data URI would bury the text; the JSON export keeps them). */
export function chatMarkdown(messages: ChatMessage[], model: string, when: Date, labels: ChatLabels): string {
  const parts = [`# colibrì: ${model}`, when.toLocaleString()]
  for (const item of messages) {
    if (item.role === "system") continue
    const lines = [`## ${item.role === "user" ? labels.user : labels.assistant}`]
    const pictures = (item.images?.length ?? 0) + (item.generated?.url ? 1 : 0)
    for (let n = 1; n <= pictures; n++) lines.push(`[${labels.image} ${n}]`)
    if (item.reasoning?.trim())
      lines.push(`<details><summary>${labels.reasoning}</summary>\n\n${item.reasoning.trim()}\n\n</details>`)
    if (item.content.trim()) lines.push(item.content.trim())
    parts.push(lines.join("\n\n"))
  }
  return parts.join("\n\n") + "\n"
}

/* A file name for an export: the date and time it was made, so two exports do not
   overwrite each other in the downloads folder. */
export function exportName(when: Date, extension: string): string {
  const pad = (n: number) => String(n).padStart(2, "0")
  return `colibri-chat-${when.getFullYear()}-${pad(when.getMonth() + 1)}-${pad(when.getDate())}-` +
    `${pad(when.getHours())}${pad(when.getMinutes())}.${extension}`
}
