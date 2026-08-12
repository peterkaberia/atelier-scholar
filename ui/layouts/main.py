from dash import html, dcc
from .sidebar import layout_sidebar, layout_mobile_topbar, layout_mobile_backdrop

# ==========================================
# 1. TAILWIND CONFIG & CUSTOM CSS INJECTION
# ==========================================
# This injects Tailwind CSS, custom fonts (Manrope & Newsreader), and Material Symbols.
# It also registers the clientside JavaScript function for textarea auto-resizing.
index_string = '''
<!DOCTYPE html>
<html class="light" lang="en">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script src="https://cdn.tailwindcss.com?plugins=forms,typography,container-queries"></script>
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
        <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,200..800;1,6..72,200..800&display=swap" rel="stylesheet"/>
        
        <script>
            tailwind.config = {
                darkMode: "class",
                theme: {
                    extend: {
                        colors: {
                            primary: "#1A237E",
                            accent: "#3B82F6",
                            "background-light": "#FDFDFD",
                            "surface-light": "#FFFFFF",
                            "border-light": "#E2E8F0",
                        },
                        fontFamily: { 
                            sans: ["Manrope", "sans-serif"],
                            serif: ["Newsreader", "serif"] 
                        },
                    },
                },
            };

            // Register Clientside Callback for Auto-resizing the Textarea
            window.dash_clientside = Object.assign({}, window.dash_clientside, {
                ui: {
                    resizeTextarea: function(value) {
                        var el = document.getElementById('search-input');
                        if(el) {
                            el.style.height = 'auto';
                            el.style.height = Math.min(el.scrollHeight, 120) + 'px';
                        }
                        return window.dash_clientside.no_update;
                    },
                    // Slides the mobile drawer sidebar in/out and toggles its
                    // backdrop. Fired by either the hamburger button (open)
                    // or the drawer's own close button / backdrop tap
                    // (close) - all three share this one toggle.
                    toggleMobileMenu: function(openClicks, closeClicks, backdropClicks) {
                        var sidebar = document.getElementById('app-sidebar');
                        var backdrop = document.getElementById('mobile-menu-backdrop');
                        if (sidebar) {
                            sidebar.classList.toggle('-translate-x-full');
                            sidebar.classList.toggle('translate-x-0');
                        }
                        if (backdrop) {
                            backdrop.classList.toggle('hidden');
                        }
                        return window.dash_clientside.no_update;
                    },
                    // Force the mobile drawer closed on every navigation, so
                    // tapping a nav link doesn't leave it open over the new
                    // page. (The Settings "Add Provider" dialogs don't need
                    // an equivalent: their open/closed state lives in a
                    // dcc.Store scoped to the /settings page content, so a
                    // fresh navigation there always starts closed with no
                    // reset needed - see ui/callbacks/settings.py.)
                    closeMobileMenuOnNav: function(pathname) {
                        var sidebar = document.getElementById('app-sidebar');
                        var backdrop = document.getElementById('mobile-menu-backdrop');
                        if (sidebar) {
                            sidebar.classList.add('-translate-x-full');
                            sidebar.classList.remove('translate-x-0');
                        }
                        if (backdrop) {
                            backdrop.classList.add('hidden');
                        }
                        return window.dash_clientside.no_update;
                    }
                }
            });

            // Inline citation hover popups for the synthesis body
            // (ui/layouts/feed.py's build_synthesis_body). The [N] markers
            // arrive as plain escaped bracket TEXT inside dcc.Markdown's
            // rendered prose (confirmed by direct inspection that embedding
            // styled HTML there and relying on dcc.Markdown to preserve it
            // does not work reliably - classes get stripped, elements get
            // restructured). So instead: after the prose renders, walk its
            // text nodes here and replace each "[N]" occurrence with a real
            // interactive <span> - this operates on the live DOM via normal
            // browser APIs, entirely bypassing dcc.Markdown's HTML-string
            // parsing/sanitization, since nothing here is fed through it.
            //
            // Citation details ride along as a data-citations JSON
            // attribute on the containing Div (a normal Dash prop, safe
            // from the markdown pipeline since it's not markdown content).

            function enrichSynthesisCitations(container) {
                if (container.dataset.citationsEnriched) return;
                // Dash/React can render this container before populating
                // dcc.Markdown's actual text into it (e.g. content still
                // streaming in from a background callback) - confirmed by
                // direct inspection that running against an empty
                // container marked itself "enriched" immediately, before
                // the real text ever arrived, permanently blocking the
                // later mutation (once text WAS there) from ever being
                // retried by this same function's own early-return guard
                // above. Bailing out without marking it lets the next
                // MutationObserver tick (see initCitationWatcher) retry
                // once there's actually something to process.
                if (!container.textContent || !container.textContent.trim()) return;
                var citations;
                try { citations = JSON.parse(container.dataset.citations || '[]'); } catch (e) { return; }
                if (!citations.length) { container.dataset.citationsEnriched = 'true'; return; }

                // React/remark renders "[1]" as separate adjacent text
                // nodes ("[", "1", "]") rather than one contiguous run -
                // confirmed by direct inspection. normalize() merges
                // adjacent text nodes back into one per run, which is what
                // lets a single regex match "[1]" as a whole below.
                container.normalize();

                var walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
                var textNodes = [];
                var node;
                while ((node = walker.nextNode())) textNodes.push(node);

                textNodes.forEach(function (textNode) {
                    var text = textNode.textContent;
                    var regex = /\[(\d+)\]/g;
                    if (!regex.test(text)) return;
                    regex.lastIndex = 0;

                    var frag = document.createDocumentFragment();
                    var lastIndex = 0;
                    var match;
                    while ((match = regex.exec(text)) !== null) {
                        if (match.index > lastIndex) {
                            frag.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
                        }
                        var marker = document.createElement('span');
                        marker.className = 'ref-badge-inline text-primary font-bold cursor-help hover:underline';
                        marker.textContent = '[' + match[1] + ']';
                        marker.dataset.idx = match[1];
                        frag.appendChild(marker);
                        lastIndex = match.index + match[0].length;
                    }
                    if (lastIndex < text.length) {
                        frag.appendChild(document.createTextNode(text.slice(lastIndex)));
                    }
                    textNode.parentNode.replaceChild(frag, textNode);
                });

                container.dataset.citationsEnriched = 'true';
            }

            // Keeps only the LATEST turn's Results/Evidence-Library section
            // (ui/layouts/feed.py's build_paper_cards, .results-accordion-
            // body) expanded, and follows the page down to whichever
            // flow-block is currently active - a new turn just appended, or
            // an in-progress one whose status text is still updating in
            // place (run_search's report()/run_investigation's report()
            // closures re-render the SAME last block repeatedly rather than
            // appending a new one per stage).
            //
            // Purely count-of-.flow-block-driven, not per-mutation: the
            // mutation observer below also fires for unrelated changes
            // (checkbox clicks, citation enrichment, dropdown menus...) -
            // gating the accordion re-collapse on "did the number of turns
            // actually change" is what lets a user's own manual toggle (see
            // the click handler further down) stick between turns instead
            // of being fought on every unrelated re-render.
            var lastFlowBlockCount = null;
            var flowScrollDebounceTimer = null;
            function enforceActiveFlowBlock() {
                var container = document.getElementById('flow-container');
                if (!container) return;
                var blocks = container.querySelectorAll(':scope > .flow-block');
                if (!blocks.length) return;

                var isFirstRun = lastFlowBlockCount === null;
                var countChanged = blocks.length !== lastFlowBlockCount;
                // Near the bottom already (e.g. actively watching a
                // multi-stage progress update stream in) - keep following
                // it. Not near the bottom - the user has deliberately
                // scrolled up to read an earlier turn, so don't yank them
                // back down just because that earlier turn's DOM mutated.
                var wasNearBottom = (container.scrollHeight - container.scrollTop - container.clientHeight) < 250;

                if (countChanged) {
                    blocks.forEach(function (block, i) {
                        var isLast = i === blocks.length - 1;
                        var body = block.querySelector('.results-accordion-body');
                        var chevron = block.querySelector('.accordion-chevron');
                        if (body) body.classList.toggle('hidden', !isLast);
                        if (chevron) chevron.classList.toggle('rotate-180', !isLast);
                    });
                    lastFlowBlockCount = blocks.length;
                }

                // Skip the very first call (initial page load/reload) -
                // landing pre-scrolled with an animated jump the instant
                // the page appears reads as a glitch, not a feature; the
                // accordion state above still gets set correctly either way.
                if (isFirstRun) return;
                if (!countChanged && !wasNearBottom) return;

                // Debounced with a real timer, not just coalesced into one
                // scroll per animation frame - confirmed live that a single
                // freshly-rendered answer fires a BURST of individual
                // mutations (enrichSynthesisCitations above replaces one
                // text node per [N] marker, one mutation each), and issuing
                // a fresh scrollIntoView for every one of them repeatedly
                // interrupted the previous call's still-in-flight smooth
                // animation - visually indistinguishable from the page
                // randomly jumping. Waiting for the burst to go quiet for
                // 250ms and firing exactly once fixes that; a later
                // qualifying mutation (e.g. the next stage's progress text)
                // still resets and fires its own scroll in turn.
                var targetBlock = blocks[blocks.length - 1];
                var scrollToStart = countChanged;
                if (flowScrollDebounceTimer) clearTimeout(flowScrollDebounceTimer);
                flowScrollDebounceTimer = setTimeout(function () {
                    flowScrollDebounceTimer = null;
                    targetBlock.scrollIntoView({ behavior: 'smooth', block: scrollToStart ? 'start' : 'end' });
                }, 250);
            }

            // True only when at least one mutation in this batch actually
            // touched #flow-container's own subtree - the observer below is
            // necessarily attached to document.body (see its own comment:
            // #flow-container doesn't exist yet when this script first
            // runs, and gets fully replaced on every route navigation), so
            // without this check, completely unrelated activity ANYWHERE on
            // the page - the sidebar's 3-second status poll
            // (ui/callbacks/ui_extras.py's update_sidebar_history), the
            // citation hover popup's one-time creation, a Settings dialog -
            // was triggering the exact same "near bottom -> scroll" path
            // above and jumping the feed out from under a user who wasn't
            // even touching it (confirmed as the dominant cause of the
            // reported random jumping, separate from the burst-mutation
            // issue the debounce above fixes).
            function touchesFlowContainer(mutationsList) {
                for (var i = 0; i < mutationsList.length; i++) {
                    var t = mutationsList[i].target;
                    if (t && t.nodeType === 1 && t.closest('#flow-container')) return true;
                }
                return false;
            }

            // Manual override for the accordion above - a click anywhere on
            // a flow-block's results-accordion-toggle (ui/layouts/feed.py's
            // build_paper_cards) flips just that block's body, independent
            // of enforceActiveFlowBlock's count-based auto-collapse.
            document.addEventListener('click', function (e) {
                var toggle = e.target.closest('.results-accordion-toggle');
                if (!toggle) return;
                var block = toggle.closest('.flow-block');
                var body = block ? block.querySelector('.results-accordion-body') : null;
                var chevron = toggle.querySelector('.accordion-chevron');
                if (!body) return;
                body.classList.toggle('hidden');
                if (chevron) chevron.classList.toggle('rotate-180');
            });

            // Runs on every DOM mutation, so it catches synthesis content
            // however it arrives (initial render, background-callback
            // update, chat history replay) without needing its own trigger.
            //
            // This whole <script> block runs while the parser is still
            // inside <head> - document.body doesn't exist yet at this
            // point, so observe(document.body, ...) would throw
            // immediately and never actually start observing anything
            // (confirmed by direct inspection: citations never got
            // enriched on a real page load, only when triggered manually
            // after the page had already fully loaded). Deferred to
            // DOMContentLoaded, with an initial scan alongside it in case
            // Dash has already rendered synthesis content by then.
            function initCitationWatcher() {
                document.querySelectorAll('.synthesis-citations:not([data-citations-enriched])').forEach(enrichSynthesisCitations);
                enforceActiveFlowBlock();
                var citationObserver = new MutationObserver(function (mutationsList) {
                    document.querySelectorAll('.synthesis-citations:not([data-citations-enriched])').forEach(enrichSynthesisCitations);
                    if (touchesFlowContainer(mutationsList)) enforceActiveFlowBlock();
                });
                citationObserver.observe(document.body, { childList: true, subtree: true });
            }
            if (document.body) {
                initCitationWatcher();
            } else {
                document.addEventListener('DOMContentLoaded', initCitationWatcher);
            }

            // One shared floating popup element, repositioned per-hover
            // rather than nested inside each marker - the marker can land
            // anywhere inside a long paragraph, so an absolutely
            // -positioned sibling wouldn't reliably have room; a
            // viewport-fixed element positioned via getBoundingClientRect
            // does.
            var citationPopupEl = null;
            function getCitationPopup() {
                if (!citationPopupEl) {
                    citationPopupEl = document.createElement('div');
                    citationPopupEl.className = 'fixed z-[9999] w-72 p-4 glass-chat-bar text-slate-900 text-xs rounded-2xl shadow-2xl pointer-events-none flex flex-col gap-1.5';
                    citationPopupEl.style.visibility = 'hidden';
                    citationPopupEl.style.opacity = '0';
                    citationPopupEl.style.transition = 'opacity 150ms';
                    document.body.appendChild(citationPopupEl);
                }
                return citationPopupEl;
            }

            function setCitationPopupContent(popup, c) {
                // Built via DOM methods (textContent), not innerHTML - c.title
                // /c.snippet come from LLM-extracted paper data and should
                // never be interpreted as markup.
                popup.innerHTML = '';

                var header = document.createElement('div');
                header.className = 'flex justify-between items-start gap-2';
                var badge = document.createElement('span');
                badge.className = 'bg-primary/10 text-primary px-2 py-0.5 rounded text-[10px] font-bold uppercase shrink-0';
                badge.textContent = 'SOURCE ' + c.index;
                var meta = document.createElement('span');
                meta.className = 'text-slate-400 italic text-[10px] text-right';
                meta.textContent = c.year + (c.journal ? ' • ' + c.journal : '');
                header.appendChild(badge);
                header.appendChild(meta);

                var titleEl = document.createElement('strong');
                titleEl.className = 'text-sm leading-snug font-bold text-primary line-clamp-2';
                titleEl.textContent = c.title;

                var snippetEl = document.createElement('span');
                snippetEl.className = 'text-slate-500 line-clamp-3 font-normal';
                snippetEl.textContent = c.snippet;

                popup.appendChild(header);
                popup.appendChild(titleEl);
                if (c.snippet) popup.appendChild(snippetEl);
            }

            document.addEventListener('mouseover', function (e) {
                var marker = e.target.closest('.ref-badge-inline');
                if (!marker) return;
                var container = marker.closest('.synthesis-citations');
                if (!container) return;
                var citations;
                try { citations = JSON.parse(container.dataset.citations || '[]'); } catch (err) { return; }
                var c = citations.find(function (x) { return String(x.index) === marker.dataset.idx; });
                if (!c) return;

                var popup = getCitationPopup();
                setCitationPopupContent(popup, c);

                var rect = marker.getBoundingClientRect();
                popup.style.left = rect.left + 'px';
                popup.style.top = (rect.top - 8) + 'px';
                popup.style.transform = 'translateY(-100%)';
                popup.style.visibility = 'visible';
                popup.style.opacity = '1';
            });
            document.addEventListener('mouseout', function (e) {
                var marker = e.target.closest('.ref-badge-inline');
                if (!marker) return;
                var popup = getCitationPopup();
                popup.style.visibility = 'hidden';
                popup.style.opacity = '0';
            });

            // Study-type filter chips (ui/layouts/feed.py's build_paper_cards).
            // Plain click delegation, not a Dash callback - these chips are
            // pure client-side show/hide over cards Dash already rendered,
            // and a session can have several independent paper-card lists
            // on one page (original search + follow-ups), so lookups are
            // scoped to the clicked chip's nearest ancestor rather than a
            // page-wide id (which would collide across flow-blocks).
            document.addEventListener('click', function (e) {
                var chip = e.target.closest('.filter-chip-btn');
                if (!chip) return;
                var scope = chip.closest('.space-y-8');
                var list = scope ? scope.querySelector('.paper-cards-list') : null;
                if (!list) return;

                var chipGroup = chip.parentElement;
                if (chipGroup) {
                    Array.prototype.forEach.call(chipGroup.querySelectorAll('.filter-chip-btn'), function (btn) {
                        var isActive = btn === chip;
                        btn.classList.toggle('active', isActive);
                        btn.classList.toggle('bg-primary', isActive);
                        btn.classList.toggle('text-white', isActive);
                        btn.classList.toggle('border-primary', isActive);
                        btn.classList.toggle('bg-white', !isActive);
                        btn.classList.toggle('text-slate-600', !isActive);
                        btn.classList.toggle('border-slate-200', !isActive);
                    });
                }

                var wantType = chip.dataset.filterType;
                Array.prototype.forEach.call(list.querySelectorAll('.atelier-result'), function (card) {
                    var show = wantType === '__all__' || card.dataset.studyType === wantType;
                    card.style.display = show ? '' : 'none';
                });
            });

            // In-app PDF viewer (ui/layouts/feed.py's .pdf-viewer-btn) -
            // opens the PDF in an iframe inside the app instead of
            // navigating away (target=_blank) or replacing the page
            // (target=_self). Pure client-side state, no Dash callback:
            // which PDF to show never needs the server.
            function closePdfViewer() {
                var modal = document.getElementById('pdf-viewer-modal');
                if (!modal) return;
                modal.classList.add('hidden');
                modal.classList.remove('flex');
                var iframe = document.getElementById('pdf-viewer-iframe');
                if (iframe) iframe.src = ''; // stop the embed from loading/playing in the background once closed
            }
            document.addEventListener('click', function (e) {
                var trigger = e.target.closest('.pdf-viewer-btn');
                if (trigger) {
                    var url = trigger.dataset.pdfUrl;
                    if (!url) return;
                    var modal = document.getElementById('pdf-viewer-modal');
                    var iframe = document.getElementById('pdf-viewer-iframe');
                    var newTabLink = document.getElementById('pdf-viewer-new-tab-link');
                    if (!modal || !iframe) return;
                    iframe.src = url;
                    if (newTabLink) newTabLink.href = url;
                    modal.classList.remove('hidden');
                    modal.classList.add('flex');
                    return;
                }
                if (e.target.closest('#pdf-viewer-close-btn')) {
                    closePdfViewer();
                    return;
                }
                // Clicking the dimmed backdrop itself (not the white card) also closes it.
                if (e.target.id === 'pdf-viewer-modal') {
                    closePdfViewer();
                }
            });
            document.addEventListener('keydown', function (e) {
                if (e.key === 'Escape') closePdfViewer();
            });

            // Per-paper summary panel (ui/callbacks/library.py opens it by
            // writing to its transform style + content; closing it is pure
            // client-side, no reason to round-trip to the server just to
            // hide something already rendered).
            document.addEventListener('click', function (e) {
                if (e.target.closest('#paper-summary-close-btn')) {
                    var panel = document.getElementById('paper-summary-panel');
                    if (panel) panel.style.transform = 'translateX(100%)';
                }
            });

            // Enter submits, Shift+Enter inserts a newline - standard
            // chat-app convention (Slack/Discord/ChatGPT). Both
            // search-input (ui/layouts/feed.py's follow-up bar) and
            // hero-search-input (ui/layouts/home.py's first-question box)
            // are dcc.Textarea, which has no built-in "submit" prop the way
            // dcc.Input's n_submit does for single-line inputs - a plain
            // textarea just inserts a line break on every Enter with no
            // way to distinguish "send" from "new line" without this
            // handler.
            document.addEventListener('keydown', function (e) {
                if (e.key !== 'Enter' || e.shiftKey) return;
                var el = e.target;
                if (!el || (el.id !== 'search-input' && el.id !== 'hero-search-input')) return;
                var btnId = el.id === 'search-input' ? 'search-btn' : 'hero-search-btn';
                var btn = document.getElementById(btnId);
                if (!btn || btn.disabled) return;
                e.preventDefault(); // stop the textarea's own default newline insertion
                btn.click();
            });
        </script>

        <style>
            /* Custom Scrollbar */
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
            
            body { font-family: 'Manrope', sans-serif; background-color: #FDFDFD; color: #0F172A; }
            .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; vertical-align: middle;}
            
            /* Home Page Effects */
            .dot-grid {
                background-image: radial-gradient(rgba(26, 35, 126, 0.08) 1.2px, transparent 0);
                background-size: 32px 32px;
            }
            .glass-panel { background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); }
            .scholar-gradient { background: linear-gradient(135deg, #1A237E 0%, #121858 100%); }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .animate-fade-in { animation: fadeIn 0.8s ease-out forwards; opacity: 0; }
            
            /* Feed & Chat Effects */
            .glass-chat-bar {
                background: rgba(255, 255, 255, 0.75);
                backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
                border: 1.5px solid #E2E8F0;
                box-shadow: 0 20px 50px -12px rgba(26, 35, 126, 0.12);
            }
            .no-scrollbar::-webkit-scrollbar { display: none; }
            .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
            .atelier-result { border-bottom: 1px solid #E2E8F0; transition: background-color 0.2s ease; }
            .atelier-result:hover { background-color: rgba(26, 35, 126, 0.02); }
            
            /* Reset Input Focus Rings */
            textarea:focus, input:focus, .Select-control:focus { outline: none !important; box-shadow: none !important; border-color: transparent !important; }

            /* Advanced AI Markdown Styling (.prose) */
            .prose h3 { font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 1rem; margin-top: 1.5rem; }
            .prose p { color: #334155; font-weight: 500; font-size: 0.95rem; line-height: 1.6; }
            .prose ul { list-style: none; padding-left: 0; }
            /* position:relative + absolutely-positioned ::before icon, NOT
               display:flex on the <li> itself - flex would treat EVERY
               direct child as its own flex item, including a <strong>
               bold label followed by a plain text node (a routine markdown
               pattern: "- **Label**: description [1]"), laying the bold
               run and the rest of the sentence out as separate columns
               instead of one wrapping paragraph. Confirmed live: exactly
               this pattern misaligned an LLM-generated recommendations
               list. Normal block flow (no flex) lets inline content -
               bold, plain text, citation badges - wrap together correctly
               regardless of how many separate inline nodes make it up. */
            .prose li { position: relative; padding-left: 2rem; margin-bottom: 1rem; color: #334155; font-weight: 500; }
            .prose li::before { content: '\\e86c'; font-family: 'Material Symbols Outlined'; color: #1A237E; font-size: 1.25rem; position: absolute; left: 0; top: -0.1rem; line-height: 1; }
            .prose table { min-width: 100%; border: 1px solid #F1F5F9; border-radius: 1rem; border-collapse: separate; border-spacing: 0; overflow: hidden; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); margin-top: 1rem; }
            .prose th { background-color: rgba(248, 250, 252, 0.5); padding: 1rem 1.5rem; text-align: left; font-weight: 700; color: #1A237E; text-transform: uppercase; font-size: 0.625rem; letter-spacing: 0.1em; border-bottom: 1px solid #F1F5F9; }
            .prose td { padding: 1.25rem 1.5rem; font-weight: 500; color: #64748B; border-bottom: 1px solid rgba(248, 250, 252, 0.5); font-size: 0.875rem;}
            
            /* Citation Tooltips */
            .citation-trigger:hover .citation-popup { opacity: 1; visibility: visible; transform: translateY(0); }
            
            /* Radix Dropdown Overrides (Tailoring Dash to match the UI template) */
            .dash-dropdown-wrapper { background-color: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; min-height: auto !important; }
            .dash-dropdown-trigger { padding: 0 !important; min-height: 0 !important; gap: 0.5rem !important;}
            .dash-dropdown-value, .dash-dropdown-value-item { display: inline-flex !important; align-items: center !important; flex: 0 0 auto !important; width: max-content !important; margin: 0 !important; padding: 0 !important; }
            .dash-dropdown-trigger-icon { display: none !important; }
            .dash-dropdown-trigger::after { content: "\\e5cf"; font-family: 'Material Symbols Outlined'; font-size: 16px !important; color: #94A3B8 !important; transition: color 0.2s ease; display: flex; align-items: center; }
            .group:hover .dash-dropdown-trigger::after { color: #1A237E !important; }
            
            .dash-dropdown-menu { background-color: white !important; border-radius: 0.75rem !important; border: 1px solid #E2E8F0 !important; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1) !important; padding: 4px !important; max-height: 320px !important; overflow-y: auto !important; }
            .dash-dropdown-item { font-size: 11px !important; font-weight: 700 !important; color: #475569 !important; padding: 8px 12px !important; border-radius: 0.5rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
            /* Rich (icon + two-line) dropdown options, e.g. Settings' provider pickers, opt out of the compact uppercase style above - it's tuned for short single-line model names, not these. */
            .dash-dropdown-item:has(.dash-dropdown-option-rich) { text-transform: none !important; letter-spacing: normal !important; padding: 4px 12px !important; }
            .dash-dropdown-option-rich { text-transform: none; }
            .dash-dropdown-item:hover, .dash-dropdown-item[data-highlighted] { background-color: #F8FAFC !important; color: #1A237E !important; }
        </style>
    </head>
    <body class="bg-background-light text-slate-900 font-sans antialiased">
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

def layout_404(message="We couldn't find the page you're looking for."):
    """Route: Fallback - Displayed when an invalid URL or ID is entered."""
    return html.Div(className="flex-1 flex flex-col items-center justify-center w-full h-full pb-32 px-8 bg-[#FDFDFD] dot-grid", children=[
        html.Span("search_off", className="material-symbols-outlined text-6xl text-slate-300 mb-6"),
        html.H1("404", className="text-4xl md:text-6xl font-extrabold tracking-tighter text-slate-900 font-sans mb-4 animate-fade-in"),
        html.P(message, className="text-xl font-serif italic text-slate-500 mb-8 animate-fade-in", style={"animationDelay": "0.1s"}),
        dcc.Link(href="/", className="animate-fade-in", style={"animationDelay": "0.2s"}, children=[
            html.Button(className="scholar-gradient text-white px-8 py-3 rounded-xl flex items-center justify-center gap-2 hover:opacity-90 active:scale-95 transition-all shadow-lg shadow-primary/20", children=[
                html.Span("arrow_back", className="material-symbols-outlined text-sm"),
                html.Span("Return Home", className="font-bold text-sm tracking-widest uppercase")
            ])
        ])
    ])

def layout_no_llm():
    """Route: Fallback - Displayed when no API keys are detected."""
    return html.Div(className="flex-1 flex flex-col items-center justify-center w-full h-full pb-32 px-8 bg-[#FDFDFD] dot-grid", children=[
        html.Span("key_off", className="material-symbols-outlined text-6xl text-red-400 mb-6 animate-fade-in"),
        html.H1("System Offline", className="text-4xl font-extrabold tracking-tighter text-slate-900 font-sans mb-4 animate-fade-in"),
        html.P("No LLM API keys were detected.", className="text-xl font-serif italic text-slate-500 mb-8 max-w-lg text-center animate-fade-in", style={"animationDelay": "0.1s"}),

        html.Div(className="bg-red-50 border border-red-100 p-6 rounded-2xl max-w-xl text-left shadow-sm animate-fade-in", style={"animationDelay": "0.2s"}, children=[
            html.H3("How to fix this:", className="text-sm font-bold text-red-900 uppercase tracking-widest mb-4"),
            html.Ul(className="list-disc pl-5 space-y-2 text-sm text-red-800 font-medium", children=[
                html.Li(["Add at least one API key in ", html.A("Settings", href="/settings", className="underline font-bold"), " (recommended - takes effect immediately, no restart)."]),
                html.Li("Or add it to your .env file and restart the server."),
            ])
        ]),

        dcc.Link(href="/settings", className="mt-8 animate-fade-in", style={"animationDelay": "0.3s"}, children=[
            html.Button(className="scholar-gradient text-white px-8 py-3 rounded-xl flex items-center justify-center gap-2 hover:opacity-90 active:scale-95 transition-all shadow-lg shadow-primary/20", children=[
                html.Span("key", className="material-symbols-outlined text-sm"),
                html.Span("Open Settings", className="font-bold text-sm tracking-widest uppercase")
            ])
        ])
    ])

def serve_layout():
    return html.Div(className="h-screen flex flex-col md:flex-row overflow-hidden antialiased", children=[
        dcc.Location(id='url', refresh=False),

        # PER-TAB WORKING STATE
        # Deliberately storage_type='memory' (Dash's default - tied to this
        # tab's own React tree only), NOT 'local'. 'local' backs onto
        # window.localStorage, which is shared across every browser tab of
        # the same origin - so two sessions running concurrently (e.g. one
        # per tab) would both read/write the same global keys and clobber
        # each other's title/records/synthesis mid-run. None of these need
        # to survive a reload: a fresh page load reconstructs everything
        # that matters (chat history, status) straight from the database via
        # layout_feed()/poll_session_status, not from these stores.
        dcc.Store(id='store-processed-records', data=[]),
        dcc.Store(id='store-current-topic', data=""),
        dcc.Store(id='store-chat-history', data=[]),
        dcc.Store(id='store-synthesis-markdown', data=""),
        dcc.Store(id='store-pending-search', storage_type='session'),

        # Gate for every background-callback write that touches visible
        # feed content (flow-container) or follow-up context (topic/
        # processed records/chat history/synthesis) - route_intent,
        # run_search, generate_synth, and run_chat (ui/callbacks/chat.py,
        # search.py) all target THIS store instead of writing those
        # directly, since a background job keeps running (and eventually
        # writes its result) even after the user has navigated to a
        # DIFFERENT session's page in the same tab - Dash Outputs target
        # component IDs, not "whichever session the user is currently
        # looking at", so an unguarded direct write would silently overwrite
        # whatever the user navigated to with a stale, unrelated session's
        # content. ui/callbacks/ui_extras.py's gate_flow_update is the one
        # place that actually forwards this to the real stores, and only
        # does so if the payload's session_id still matches
        # current-session-id's LIVE value at that moment.
        dcc.Store(id='pending-flow-update'),

        # FEED LOGIC TRIGGERS
        dcc.Store(id='trigger-router', data=""),
        dcc.Store(id='trigger-search', data=""),
        dcc.Store(id='trigger-synthesis', data=""),
        dcc.Store(id='trigger-chat', data=""),
        dcc.Store(id='trigger-investigate', data=""),

        # MOBILE NAV TRIGGERS (see toggleMobileMenu/closeMobileMenuOnNav in app.py)
        dcc.Store(id='mobile-menu-toggle-dummy', data=0),
        dcc.Store(id='mobile-menu-nav-dummy', data=0),

        # MOBILE TOP BAR - the sidebar's replacement when it's off-screen (<md)
        layout_mobile_topbar(),

        # MOBILE DRAWER BACKDROP
        layout_mobile_backdrop(),

        # SIDEBAR (static column on desktop, slide-in drawer on mobile)
        layout_sidebar(),

        # MAIN ROUTING CANVAS
        html.Main(id="page-content", className="flex-1 flex flex-col relative h-full bg-[#FDFDFD] overflow-hidden"),

        # PDF VIEWER MODAL - opened/closed entirely client-side (JS click
        # delegation in this file's index_string, see .pdf-viewer-btn):
        # which PDF to show is pure ephemeral UI state with no server round
        # trip needed, unlike the paper-summary panel below which needs an
        # LLM call. Ships hidden/empty; JS sets the iframe src and toggles
        # visibility. The "open in new tab" link is a permanent fallback,
        # not conditional - iframes can't reliably tell JS when a publisher
        # blocks embedding via X-Frame-Options, so rather than guess, the
        # escape hatch is always visible.
        # z-[350]: higher than paper-summary-panel's z-[300] below - the PDF
        # viewer can be opened FROM WITHIN that panel (its own "View PDF"
        # button) and must stack above it, not behind it. Both are well
        # above the chat bar's z-[100] and the sidebar's z-[200] - they
        # previously used the Tailwind default z-50, which put them BEHIND
        # both (confirmed live: the popups rendered under the bottom chat
        # bar and the sidebar).
        html.Div(id='pdf-viewer-modal', className="fixed inset-0 z-[350] hidden items-center justify-center bg-black/60 backdrop-blur-sm p-4", children=[
            html.Div(className="bg-white rounded-2xl shadow-2xl w-full h-full max-w-5xl flex flex-col overflow-hidden", children=[
                html.Div(className="flex items-center justify-between px-5 py-3 border-b border-slate-200 flex-shrink-0", children=[
                    html.Span("PDF Viewer", className="font-bold text-slate-700 text-sm"),
                    html.Div(className="flex items-center gap-5", children=[
                        html.A("Open in new tab ↗", id='pdf-viewer-new-tab-link', href="#", target="_blank", className="text-xs font-bold text-primary hover:underline"),
                        html.Button(html.Span("close", className="material-symbols-outlined"), id='pdf-viewer-close-btn', className="text-slate-400 hover:text-slate-900 flex items-center")
                    ])
                ]),
                html.Iframe(id='pdf-viewer-iframe', src="", className="flex-1 w-full border-0")
            ])
        ]),

        # PER-PAPER SUMMARY PANEL - opened by a Dash callback
        # (ui/callbacks/library.py's summarize_selected_paper, since it
        # needs a server-side LLM call), closed purely client-side (no
        # server round trip needed just to hide it again).
        html.Div(id='paper-summary-panel', className="fixed top-0 right-0 z-[300] h-full w-full md:w-[420px] bg-white shadow-2xl border-l border-slate-200 flex flex-col", style={"transform": "translateX(100%)", "transition": "transform 0.25s ease-out"}, children=[
            html.Div(className="flex items-center justify-between px-5 py-4 border-b border-slate-200 flex-shrink-0", children=[
                html.Span("Paper Summary", className="font-bold text-slate-700 text-sm uppercase tracking-widest"),
                html.Button(html.Span("close", className="material-symbols-outlined"), id='paper-summary-close-btn', className="text-slate-400 hover:text-slate-900 flex items-center")
            ]),
            # flex-1 min-h-0 lives on dcc.Loading itself, not just the inner
            # content div - dcc.Loading renders its own wrapper div around
            # its child, so the PANEL's flex-col layout was sizing THAT
            # wrapper (which had no flex-1 of its own) to its content's
            # natural height instead of constraining it to the remaining
            # panel space. With no bounded height anywhere in the chain,
            # the inner div's overflow-y-auto had nothing to actually
            # engage against - content just overflowed the panel with no
            # scrollbar (confirmed live). min-h-0 overrides a flex item's
            # default min-height:auto, which otherwise refuses to shrink
            # below its content size even with flex-1 set - the other half
            # of the same overflow-in-flex gotcha.
            dcc.Loading(
                html.Div(id='paper-summary-content', className="h-full overflow-y-auto p-5 text-sm text-slate-700 leading-relaxed space-y-3"),
                # parent_className (not className) - it's the one that
                # lands on dcc.Loading's OUTERMOST wrapper div (confirmed
                # via dash's own component metadata: className only reaches
                # an inner "root DOM node", not what the panel's flex-col
                # layout actually sizes as this child's box).
                parent_className="flex-1 min-h-0",
            ),
        ]),

        dcc.Download(id='download-references'),
        dcc.Download(id='download-pdf'),
    ])