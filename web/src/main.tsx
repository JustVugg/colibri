import { createRoot } from "react-dom/client"

import App from "./App"
import { ErrorBoundary } from "./ErrorBoundary"
import { LocaleProvider } from "./i18n"
import "./index.css"

createRoot(document.getElementById("root")!).render(
  <ErrorBoundary>
    <LocaleProvider>
      <App />
    </LocaleProvider>
  </ErrorBoundary>,
)
