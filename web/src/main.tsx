import { createRoot } from "react-dom/client"

import App from "./App"
import { ErrorBoundary } from "./ErrorBoundary"
import "./index.css"

createRoot(document.getElementById("root")!).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>,
)
