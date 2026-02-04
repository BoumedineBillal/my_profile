const projects = [
    {
        id: "yolo26n-esp32p4",
        title: "World’s First YOLO26n on ESP32-P4",
        desc: "Adapting Training Pipelines for QAT and Graph Optimization to Achieve 36.5% mAP at 512px Resolution with 1.7s Latency",
        img: "projects/yolo26n-esp32p4/assets/yolo26n-esp32p4.jpg",
        link: "project-viewer.html?id=yolo26n-esp32p4"
    },
    {
        id: "coming_soon",
        title: "Coming Soon",
        desc: "Coming Soon",
        img: "public/assets/projects/coming_soon.jpg",
        link: "#"
    },
    {
        id: "coming_soon",
        title: "Coming Soon",
        desc: "Coming Soon",
        img: "public/assets/projects/coming_soon.jpg",
        link: "#"
    }
];

const createProjectCard = (project) => `
    <a href="${project.link}" class="group block bg-slate-800/50 rounded-xl overflow-hidden border border-slate-700/50 hover:-translate-y-2 hover:border-cyan-500 hover:shadow-[0_0_30px_-5px_rgba(6,182,212,0.3)] transition-all duration-300 h-full flex flex-col">
        <!-- Top: Image Area (Elongated Portrait 4:5 ratio) -->
        <div class="relative w-full aspect-video bg-slate-900 border-b border-slate-700/50 overflow-hidden">
            <img src="${project.img}" alt="${project.title}" class="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" onerror="this. style.display='none'; this.nextElementSibling.style.display='flex'">
            
            <!-- Fallback Placeholder -->
            <div class="hidden absolute inset-0 flex-col items-center justify-center p-6 text-center bg-slate-800/80">
                <div class="text-slate-600 mb-3">
                    <svg class="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                </div>
                <span class="font-mono text-xs text-slate-500 tracking-wider">[${project.id.toUpperCase()}.JPG]</span>
            </div>
        </div>
        
        <!-- Middle: Content -->
        <div class="p-6 flex-grow flex flex-col">
            <h3 class="text-xl font-bold text-white mb-2 group-hover:text-cyan-300 transition-colors">${project.title}</h3>
            <p class="text-slate-400 text-sm leading-relaxed">${project.desc}</p>
        </div>

        <!-- Bottom: Indicator -->
        <div class="px-6 py-4 border-t border-slate-700/50 flex items-center justify-between text-cyan-500/80 text-xs font-mono group-hover:bg-cyan-500/5 transition-colors">
            <span>// VIEW_MODULE</span>
            <svg class="w-4 h-4 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"></path></svg>
        </div>
    </a>
`;

document.addEventListener('DOMContentLoaded', () => {
    const gridContainer = document.getElementById('project-grid');
    if (gridContainer) {
        gridContainer.innerHTML = projects.map(createProjectCard).join('');
    }
});
