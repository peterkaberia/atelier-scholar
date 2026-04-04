from dash import html, dcc
from .sidebar import layout_sidebar

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
                    }
                }
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
            .prose li { display: flex; align-items: flex-start; margin-bottom: 1rem; color: #334155; font-weight: 500; }
            .prose li::before { content: '\\e86c'; font-family: 'Material Symbols Outlined'; color: #1A237E; font-size: 1.25rem; margin-right: 0.75rem; margin-top: -0.1rem; }
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
            
            .dash-dropdown-menu { background-color: white !important; border-radius: 0.75rem !important; border: 1px solid #E2E8F0 !important; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1) !important; padding: 4px !important; }
            .dash-dropdown-item { font-size: 11px !important; font-weight: 700 !important; color: #475569 !important; padding: 8px 12px !important; border-radius: 0.5rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
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
        html.P("No LLM API keys were detected in your environment.", className="text-xl font-serif italic text-slate-500 mb-8 max-w-lg text-center animate-fade-in", style={"animationDelay": "0.1s"}),
        
        html.Div(className="bg-red-50 border border-red-100 p-6 rounded-2xl max-w-xl text-left shadow-sm animate-fade-in", style={"animationDelay": "0.2s"}, children=[
            html.H3("How to fix this:", className="text-sm font-bold text-red-900 uppercase tracking-widest mb-4"),
            html.Ul(className="list-disc pl-5 space-y-2 text-sm text-red-800 font-medium", children=[
                html.Li("Create or check your .env file in the project root."),
                html.Li("Ensure you have added at least one valid API key (e.g., GROQ_API_KEY=...)."),
                html.Li("Restart your Python server so the keys load into the environment.")
            ])
        ])
    ])

def serve_layout():
    return html.Div(className="h-screen flex overflow-hidden antialiased", children=[
        dcc.Location(id='url', refresh=False),
        
        # GLOBAL STATE
        dcc.Store(id='store-all-records', data=[], storage_type='local'),
        dcc.Store(id='store-processed-records', data=[], storage_type='local'),
        dcc.Store(id='store-current-topic', data="", storage_type='local'),
        dcc.Store(id='store-chat-history', data=[], storage_type='local'),
        dcc.Store(id='store-user-history', data=[], storage_type='local'),
        dcc.Store(id='store-synthesis-markdown', data="", storage_type='local'),
        dcc.Store(id='store-pending-search', storage_type='session'),
        
        # FEED LOGIC TRIGGERS
        dcc.Store(id='trigger-router', data=""),
        dcc.Store(id='trigger-search', data=""), 
        dcc.Store(id='trigger-synthesis', data=""),
        dcc.Store(id='trigger-chat', data=""),

        # SIDEBAR
        layout_sidebar(),

        # MAIN ROUTING CANVAS
        html.Main(id="page-content", className="flex-1 flex flex-col relative h-full bg-[#FDFDFD] overflow-hidden")
    ])