import * as React from "react"

import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn("h-10 w-full rounded-lg border border-border bg-input/50 px-3 text-sm outline-none transition focus:border-primary/60 focus:ring-2 focus:ring-primary/15", className)}
      {...props}
    />
  )
}

export { Input }
