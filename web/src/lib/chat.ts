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
