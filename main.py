<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Simulador SBML</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background: #111; color: white; }
    .accordion-btn { background:#1a1a1a; font-weight:bold; }
    .accordion-btn:hover { background:#222; }
    .accordion-content { background:#0d0d0d; }
    #resizer { cursor: col-resize; }
  </style>
</head>
<body class="flex flex-col min-h-screen">

<!-- Paso 1: Hero -->
<section id="hero" class="flex-1 flex flex-col justify-center items-center text-center">
  <h1 class="text-5xl font-bold text-indigo-600">Simulador de SBML</h1>
  <p class="mt-4 text-lg text-gray-400">Sube un archivo SBML para comenzar</p>
  <input id="sbmlFile" type="file" accept=".xml,.sbml" 
         class="mt-8 px-6 py-3 bg-indigo-600 text-white font-medium rounded-lg cursor-pointer">
</section>

<!-- Paso 2: Menú acción -->
<section id="menuAction" class="hidden flex flex-col justify-center items-center text-center space-y-6">
  <h2 class="text-2xl font-semibold mb-6">¿Qué deseas hacer?</h2>
  <div class="flex gap-6">
    <button id="btnSim" class="px-6 py-3 bg-green-700 text-white rounded-lg shadow-lg hover:-translate-y-1 transform transition">Simulación</button>
    <button id="btnGraph" class="px-6 py-3 bg-yellow-500 text-white rounded-lg shadow-lg hover:-translate-y-1 transform transition">Diagrama</button>
    <button id="btnOpt" class="px-6 py-3 bg-blue-600 text-white rounded-lg shadow-lg hover:-translate-y-1 transform transition">Optimizar</button>
  </div>
</section>

<!-- Paso 3: Simulación -->
<section id="sim-view" class="hidden flex flex-col md:flex-row flex-1 bg-white relative">

  <!-- Panel lateral redimensionable -->
  <div id="panel" class="bg-black flex-shrink-0 flex flex-col border-r border-gray-700 overflow-y-auto relative" style="width:300px; min-width:200px; max-width:600px;">
    <div id="resizer" class="absolute top-0 right-0 h-full w-1 bg-gray-700 hover:bg-gray-500"></div>
    <div class="p-4 text-xl font-bold border-b border-gray-700 flex justify-between items-center">
      Opciones
      <button onclick="goBackMenu()" class="text-sm px-2 py-1 bg-gray-700 rounded hover:bg-gray-600">← Volver</button>
    </div>
    <div class="flex-1 p-4 space-y-4">

      <!-- Tiempo -->
      <div>
        <button class="accordion-btn w-full px-4 py-3 text-lg accordion-trigger">Tiempo</button>
        <div class="accordion-content p-4 space-y-2 hidden">
          <label class="block text-sm">Inicio</label>
          <input type="number" id="tStart" value="0" step="any" class="w-full p-1 bg-gray-900 text-white border border-gray-700 rounded"/>
          <label class="block text-sm">Fin</label>
          <input type="number" id="tEnd" value="50" step="any" class="w-full p-1 bg-gray-900 text-white border border-gray-700 rounded"/>
          <label class="block text-sm">Puntos</label>
          <input type="number" id="nPoints" value="200" step="1" class="w-full p-1 bg-gray-900 text-white border border-gray-700 rounded"/>
        </div>
      </div>

      <!-- Parámetros -->
      <div>
        <button class="accordion-btn w-full px-4 py-3 text-lg accordion-trigger flex justify-between items-center">
          Parámetros
          <button id="resetParams" class="text-xs px-3 py-1 bg-red-700 rounded hover:bg-red-600">Restaurar</button>
        </button>
        <div id="param-sliders" class="accordion-content p-4 grid gap-3 hidden"></div>
      </div>

      <!-- Condiciones iniciales -->
      <div>
        <button class="accordion-btn w-full px-4 py-3 text-lg accordion-trigger flex justify-between items-center">
          Condiciones iniciales
          <button id="resetInitConds" class="text-xs px-3 py-1 bg-red-700 rounded hover:bg-red-600">Restaurar</button>
        </button>
        <div id="init-conds" class="accordion-content p-4 grid gap-3 hidden"></div>
      </div>

      <!-- Especies -->
      <div>
        <button class="accordion-btn w-full px-4 py-3 text-lg accordion-trigger flex justify-between items-center">
          Especies
          <button id="uncheckAll" class="text-xs px-3 py-1 bg-gray-700 rounded hover:bg-gray-600">Desmarcar todo</button>
        </button>
        <div id="species-list" class="accordion-content p-4 grid gap-2 hidden"></div>
      </div>

    </div>
  </div>

  <!-- Gráfica -->
  <div class="flex-1 relative overflow-hidden h-screen">
    <canvas id="plot" class="w-full h-full"></canvas>
  </div>

</section>

<!-- Paso 3: Diagrama -->
<section id="graph-view" class="hidden flex flex-col flex-1 bg-white">
  <button onclick="goBackMenu()" class="px-4 py-2 bg-gray-300 m-4 rounded hover:bg-gray-400 w-fit">← Regresar</button>
  <div id="cy" class="flex-1"></div>
</section>

<!-- Paso 3: Optimizar -->
<section id="opt-view" class="hidden flex flex-col flex-1 bg-white">
  <button onclick="goBackMenu()" class="px-4 py-2 bg-gray-300 m-4 rounded hover:bg-gray-400 w-fit">← Regresar</button>
  <div class="flex-1 flex justify-center items-center">
    <img src="https://placekitten.com/400/300" class="rounded-lg shadow-lg"/>
  </div>
</section>

<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>

<script>
let sbmlFile = null;
let species = {}, parameters = {}, initialConditions = {};
let defaultParams = {}, defaultInitials = {};
let selectedSpecies = new Set();
let chart = null;
let reactions = [];
let multiplierOptions = [1, 10, 100, 1000];

// -------- Subir archivo --------
document.getElementById("sbmlFile").addEventListener("change", async (e)=>{
  sbmlFile=e.target.files[0];
  if(!sbmlFile) return;
  const fd=new FormData(); fd.append("file",sbmlFile);
  const res=await fetch("https://backend-2e5l.onrender.com/inspect",{method:"POST",body:fd});
  const data=await res.json();

  species=data.species||{};
  parameters=data.parameters||{};
  initialConditions=data.initial_conditions||{};
  reactions=data.reactions||[];
  defaultParams={...parameters};
  defaultInitials={...initialConditions};
  selectedSpecies=new Set(data.defaultSelections||Object.keys(species));

  createSliders();
  createSpeciesSelector();
  createInitialConditionsEditor();

  document.getElementById("hero").classList.add("hidden");
  document.getElementById("menuAction").classList.remove("hidden");
});

// -------- Menu acción --------
document.getElementById("btnSim").onclick = () => {
  document.getElementById("menuAction").classList.add("hidden");
  document.getElementById("sim-view").classList.remove("hidden");
  simulate();
};
document.getElementById("btnGraph").onclick = () => {
  document.getElementById("menuAction").classList.add("hidden");
  document.getElementById("graph-view").classList.remove("hidden");
  drawGraph();
};
document.getElementById("btnOpt").onclick = () => {
  document.getElementById("menuAction").classList.add("hidden");
  document.getElementById("opt-view").classList.remove("hidden");
};
function goBackMenu(){
  document.getElementById("graph-view").classList.add("hidden");
  document.getElementById("sim-view").classList.add("hidden");
  document.getElementById("opt-view").classList.add("hidden");
  document.getElementById("menuAction").classList.remove("hidden");
}

// -------- Multiplicador --------
function attachMultiplier(num, rng, obj, key){
  let multBtn=document.createElement("button");
  multBtn.textContent="x1";
  multBtn.dataset.multIndex=0;
  multBtn.className="px-2 py-1 text-xs bg-indigo-700 rounded hover:bg-indigo-600";
  multBtn.onclick=(e)=>{
    e.preventDefault();
    let idx=(parseInt(multBtn.dataset.multIndex)+1)%multiplierOptions.length;
    multBtn.dataset.multIndex=idx;
    multBtn.textContent="x"+multiplierOptions[idx];
  };

  function applyValue(val){
    obj[key]=parseFloat(val);
    num.value=obj[key];
    rng.value=obj[key];
    simulate();
  }

  num.addEventListener("keydown",(e)=>{
    let idx=parseInt(multBtn.dataset.multIndex);
    let factor=multiplierOptions[idx];
    if(e.key==="ArrowUp"){ e.preventDefault(); applyValue(parseFloat(num.value)+factor); }
    if(e.key==="ArrowDown"){ e.preventDefault(); applyValue(parseFloat(num.value)-factor); }
  });
  num.onwheel=(e)=>{
    e.preventDefault();
    let idx=parseInt(multBtn.dataset.multIndex);
    let factor=multiplierOptions[idx];
    let delta=(e.deltaY<0?1:-1)*factor;
    applyValue(parseFloat(num.value)+delta);
  };
  num.oninput=()=>applyValue(num.value);
  rng.oninput=()=>applyValue(rng.value);

  return multBtn;
}

// -------- Sliders y checkboxes --------
function createSliders(){
  const cont=document.getElementById("param-sliders"); cont.innerHTML="";
  Object.entries(parameters).forEach(([k,v])=>{
    let row=document.createElement("div"); row.className="flex items-center space-x-2";
    let label=document.createElement("label"); label.textContent=k; label.className="w-32 text-xs";
    let num=document.createElement("input"); num.type="number"; num.value=v; num.step="any"; num.className="border p-1 rounded w-20 text-xs bg-gray-900 text-white border-gray-700";
    let rng=document.createElement("input"); rng.type="range"; rng.min=v*0.1||0; rng.max=v*2||1; rng.step=(v/100)||0.01; rng.value=v; rng.className="flex-1";
    let multBtn=attachMultiplier(num,rng,parameters,k);
    row.append(label,num,rng,multBtn); cont.appendChild(row);
  });
}
function createInitialConditionsEditor(){
  const cont=document.getElementById("init-conds"); cont.innerHTML="";
  Object.entries(initialConditions).forEach(([k,v])=>{
    let row=document.createElement("div"); row.className="flex items-center space-x-2";
    let label=document.createElement("label"); label.textContent=species[k]||k; label.className="w-32 text-xs";
    let num=document.createElement("input"); num.type="number"; num.value=v; num.step="any"; num.className="border p-1 rounded w-20 text-xs bg-gray-900 text-white border-gray-700";
    let rng=document.createElement("input"); rng.type="range"; rng.min=v*0.1||0; rng.max=v*2||1; rng.step=(v/100)||0.01; rng.value=v; rng.className="flex-1";
    let multBtn=attachMultiplier(num,rng,initialConditions,k);
    row.append(label,num,rng,multBtn); cont.appendChild(row);
  });
}
function createSpeciesSelector(){
  const cont=document.getElementById("species-list"); cont.innerHTML="";
  Object.entries(species).forEach(([id,name])=>{
    let chk=document.createElement("input"); chk.type="checkbox"; chk.checked=selectedSpecies.has(id);
    chk.onchange=()=>{chk.checked?selectedSpecies.add(id):selectedSpecies.delete(id);simulate();};
    let lbl=document.createElement("label"); lbl.textContent=name; lbl.className="text-xs";
    let row=document.createElement("div"); row.className="flex items-center space-x-2"; row.append(chk,lbl); cont.appendChild(row);
  });
}

// -------- Botones de restaurar --------
document.getElementById("resetParams").onclick=()=>{parameters={...defaultParams}; createSliders(); simulate();};
document.getElementById("resetInitConds").onclick=()=>{initialConditions={...defaultInitials}; createInitialConditionsEditor(); simulate();};
document.getElementById("uncheckAll").onclick=()=>{selectedSpecies.clear(); createSpeciesSelector(); simulate();};

// -------- Simulación --------
async function simulate(){
  if(!sbmlFile) return;
  const fd=new FormData();
  fd.append("file",sbmlFile);
  fd.append("t_start",document.getElementById("tStart").value);
  fd.append("t_end",document.getElementById("tEnd").value);
  fd.append("n_points",document.getElementById("nPoints").value);
  fd.append("selected_species",[...selectedSpecies].join(","));
  fd.append("param_values_json",JSON.stringify(parameters));
  fd.append("initial_conditions_json",JSON.stringify(initialConditions));
  const res=await fetch("https://backend-2e5l.onrender.com/simulate",{method:"POST",body:fd});
  const data=await res.json();
  plotData(data);
}
function plotData(data){
  const ctx=document.getElementById("plot").getContext("2d");
  if(chart) chart.destroy();
  const labels=data.data.map(r=>r[0]);
  const datasets=data.columns.slice(1).map((col,i)=>({
    label:col, data:data.data.map(r=>r[i+1]), borderWidth:2, fill:false, tension:0.1
  }));
  chart=new Chart(ctx,{
    type:"line",
    data:{labels,datasets},
    options:{ responsive:true, maintainAspectRatio:false, animation:false, resizeDelay:0,
      scales:{x:{title:{display:true,text:"time"}}}}
  });
}

// -------- Grafo --------
function drawGraph(){
  if(!Object.keys(species).length) return;
  let elements=[];
  Object.keys(species).forEach(id=>elements.push({data:{id:id,label:species[id]}}));
  reactions.forEach(r=>{
    r.reactants.forEach(react=>elements.push({data:{source:react,target:r.id}}));
    r.products.forEach(prod=>elements.push({data:{source:r.id,target:prod}}));
    elements.push({data:{id:r.id,label:r.id},classes:"reaction"});
  });
  cytoscape({container:document.getElementById("cy"),elements:elements,style:[
    {selector:"node",style:{"label":"data(label)","text-valign":"center","color":"#fff","background-color":"#4f46e5","text-outline-color":"#4f46e5","text-outline-width":2}},
    {selector:"node.reaction",style:{"shape":"rectangle","background-color":"#f59e0b","text-outline-color":"#f59e0b","color":"#000"}},
    {selector:"edge",style:{"width":2,"line-color":"#9ca3af","target-arrow-color":"#9ca3af","target-arrow-shape":"triangle"}}
  ],layout:{name:"cose"}});
}

// -------- Acordeones --------
document.querySelectorAll(".accordion-trigger").forEach(btn=>{
  btn.addEventListener("click",()=>{
    const content=btn.parentElement.querySelector(".accordion-content");
    if(content){ content.classList.toggle("hidden"); if(chart) setTimeout(()=>chart.resize(),200); }
  });
});

// -------- Tiempo dispara simulación --------
["tStart","tEnd","nPoints"].forEach(id=>{
  document.getElementById(id).oninput=()=>simulate();
});

// -------- Resizer del panel --------
const panel=document.getElementById("panel");
const resizer=document.getElementById("resizer");
let isResizing=false;
resizer.addEventListener("mousedown",(e)=>{e.preventDefault();isResizing=true;document.body.style.cursor="col-resize";});
document.addEventListener("mousemove",(e)=>{
  if(!isResizing) return;
  let newWidth=e.clientX;
  if(newWidth<200) newWidth=200;
  if(newWidth>600) newWidth=600;
  panel.style.width=newWidth+"px";
});
document.addEventListener("mouseup",()=>{if(isResizing){isResizing=false;document.body.style.cursor="default";}});
</script>
</body>
</html>
