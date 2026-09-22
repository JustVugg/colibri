/** Original pixel mark from site/index.html; wordmark uses its Bytesized font.
 *
 * I colori sono quelli di assets/colibri-icon.svg, non inventati qui: e la
 * stessa griglia (translate(7 21), celle da 14) quindi le due immagini non
 * possono divergere per svista. Lo sfondo arrotondato di quel file NON viene
 * riportato: qui il marchio sta sulla superficie dell'app, che ha gia la sua.
 *
 * Le tinte sono fisse e non seguono currentColor: sono sature abbastanza da
 * reggere sia sul tema scuro sia sul chiaro. La parola accanto invece resta
 * in currentColor, cosi segue il tema come il resto del testo. */
export function Brand({ word = false }: { word?: boolean }) {
  return <span className={word ? "colibri-brand full" : "colibri-brand"} aria-label="colibrì">
    <svg viewBox="0 21 168 126" aria-hidden="true" stroke="none"><g shapeRendering="crispEdges" stroke="none" transform="translate(7 21)">
<rect height="14" width="14" x="56" y="0" fill="#d75fd7"></rect><rect height="14" width="14" x="70" y="0" fill="#d75fd7"></rect><rect height="14" width="14" x="84" y="0" fill="#d75fd7"></rect>
<rect height="14" width="14" x="42" y="14" fill="#d75fd7"></rect><rect height="14" width="14" x="56" y="14" fill="#d75fd7"></rect><rect height="14" width="14" x="70" y="14" fill="#d75fd7"></rect><rect height="14" width="14" x="84" y="14" fill="#d75fd7"></rect><rect height="14" width="14" x="98" y="14" fill="#d75fd7"></rect><rect height="14" width="14" x="140" y="14" fill="#5fd7d7"></rect>
<rect height="14" width="14" x="56" y="28" fill="#d75fd7"></rect><rect height="14" width="14" x="70" y="28" fill="#d75fd7"></rect><rect height="14" width="14" x="84" y="28" fill="#d75fd7"></rect><rect height="14" width="14" x="98" y="28" fill="#d75fd7"></rect><rect height="14" width="14" x="126" y="28" fill="#5fd7d7"></rect><rect height="14" width="14" x="140" y="28" fill="#5fd7d7"></rect>
<rect height="14" width="14" x="0" y="42" fill="#ff8700"></rect><rect height="14" width="14" x="14" y="42" fill="#ff8700"></rect><rect height="14" width="14" x="28" y="42" fill="#ff8700"></rect><rect height="14" width="14" x="42" y="42" fill="#ff8700"></rect><rect height="14" width="14" x="56" y="42" fill="#00afaf"></rect><rect height="14" width="14" x="70" y="42" fill="#00afaf"></rect><rect height="14" width="14" x="84" y="42" fill="#fff"></rect><rect height="14" width="14" x="98" y="42" fill="#00afaf"></rect><rect height="14" width="14" x="112" y="42" fill="#5fd7d7"></rect><rect height="14" width="14" x="126" y="42" fill="#5fd7d7"></rect>
<rect height="14" width="14" x="56" y="56" fill="#00afaf"></rect><rect height="14" width="14" x="70" y="56" fill="#00afaf"></rect><rect height="14" width="14" x="84" y="56" fill="#00afaf"></rect><rect height="14" width="14" x="98" y="56" fill="#00afaf"></rect><rect height="14" width="14" x="112" y="56" fill="#00afaf"></rect><rect height="14" width="14" x="126" y="56" fill="#5fd7d7"></rect><rect height="14" width="14" x="140" y="56" fill="#5fd7d7"></rect>
<rect height="14" width="14" x="70" y="70" fill="#00afaf"></rect><rect height="14" width="14" x="84" y="70" fill="#00afaf"></rect><rect height="14" width="14" x="98" y="70" fill="#00afaf"></rect><rect height="14" width="14" x="112" y="70" fill="#00afaf"></rect><rect height="14" width="14" x="126" y="70" fill="#5fd7d7"></rect><rect height="14" width="14" x="140" y="70" fill="#5fd7d7"></rect>
<rect height="14" width="14" x="84" y="84" fill="#00afaf"></rect><rect height="14" width="14" x="98" y="84" fill="#00afaf"></rect><rect height="14" width="14" x="112" y="84" fill="#5fd7d7"></rect><rect height="14" width="14" x="126" y="84" fill="#5fd7d7"></rect>
<rect height="14" width="14" x="98" y="98" fill="#00afaf"></rect><rect height="14" width="14" x="112" y="98" fill="#5fd7d7"></rect>
<rect height="14" width="14" x="112" y="112" fill="#5fd7d7"></rect>
</g></svg>
    {word && <span className="colibri-word">colibri</span>}
  </span>
}
