"""
Decorative page effects shared by the dashboard.
"""

import streamlit as st


def inject_wave_animation():
    """Inject animated SVG water waves fixed to the bottom of the viewport.
    Uses inline <svg> with CSS keyframe translation for a flowing wave effect."""
    # components.html (not st.html) because this needs to run JavaScript:
    # st.html strips scripts, and the waves are injected into the parent
    # document so they sit behind the whole page rather than in a box.
    import streamlit.components.v1 as components
    components.html("""
    <style>
      html, body {
        margin: 0; padding: 0;
        background: transparent !important;
        overflow: hidden;
      }
      .wave-wrap {
        position: fixed;
        bottom: 0; left: 0;
        width: 100vw; height: 35vh;
        pointer-events: none;
        z-index: 0;
        opacity: 0.40;
      }
      .wave-wrap svg {
        position: absolute;
        bottom: 0; left: 0;
        width: 200%;
        height: 100%;
      }
      .wave-svg-1 { animation: wScroll1 7s linear infinite; }
      .wave-svg-2 { animation: wScroll2 11s linear infinite; opacity: 0.6; }
      .wave-svg-3 { animation: wScroll3 15s linear infinite; opacity: 0.35; }

      @keyframes wScroll1 {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
      }
      @keyframes wScroll2 {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
      }
      @keyframes wScroll3 {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
      }
    </style>

    <div class="wave-wrap">
      <!-- Wave 1 — front -->
      <svg class="wave-svg-1" viewBox="0 0 2400 320" preserveAspectRatio="none"
           xmlns="http://www.w3.org/2000/svg">
        <path fill="#00d4ff"
              d="M0,224 C200,128 400,288 600,224 C800,160 1000,288 1200,224
                 C1400,128 1600,288 1800,224 C2000,160 2200,288 2400,224 L2400,320 L0,320 Z"/>
      </svg>
      <!-- Wave 2 — mid -->
      <svg class="wave-svg-2" viewBox="0 0 2400 320" preserveAspectRatio="none"
           xmlns="http://www.w3.org/2000/svg">
        <path fill="#3cb8de"
              d="M0,256 C200,192 400,320 600,256 C800,192 1000,320 1200,256
                 C1400,192 1600,320 1800,256 C2000,192 2200,320 2400,256 L2400,320 L0,320 Z"/>
      </svg>
      <!-- Wave 3 — back -->
      <svg class="wave-svg-3" viewBox="0 0 2400 320" preserveAspectRatio="none"
           xmlns="http://www.w3.org/2000/svg">
        <path fill="#00d4ff"
              d="M0,288 C200,224 400,320 600,288 C800,224 1000,320 1200,288
                 C1400,224 1600,320 1800,288 C2000,224 2200,320 2400,288 L2400,320 L0,320 Z"/>
      </svg>
    </div>

    <script>
      // Break out of Streamlit iframe — inject into parent document
      try {
        var parent = window.parent.document;
        var existing = parent.getElementById('fermium-waves');
        if (existing) existing.remove();

        var div = document.querySelector('.wave-wrap');
        var style = document.querySelector('style');
        var container = parent.createElement('div');
        container.id = 'fermium-waves';
        container.innerHTML = style.outerHTML + div.outerHTML;
        parent.body.appendChild(container);
      } catch(e) {}
    </script>
    """, height=0)
