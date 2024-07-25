import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import vertexShader from '../../shaders/vertex.glsl.js'
import fragmentShader from '../../shaders/fragment.glsl.js'

import {generateLegend, fontManager} from '../legend/legendMaker.js';
import {makeNodes, makeConnectivityEdges, makeTopologyEdges, setNodePos, setAllEdgePosFromNodePos, setEdgePosFromNodePos,
    computeClusterParams, colormapLinear, color1, color2, addNodesSimple, activateNodes} from '../hierarchical/graphMaker.js';

import {calcMove} from '../force/force-directed.js'
import * as data from '../../saves/net_data_medium1.json' assert {type: 'json'}; // medium1
console.log(data);

const fontPath = 'fonts/helvetiker_regular.typeface.json';

let camera, scene, renderer, stats;
let clusterGroup, clusEdges;
let edgeConnectivity, edgeTopology;
let uiScene, orthoCamera;
let ws;
let maxLabelEntity = null;

// Legend Parameters
const defWidth = 900; 
const defHeight = 500; 

// Node & Edge Parameters
const maxNodes = 100;
const clusIdx = 0; // Place holder for single cluster
const sizeMult = .5;
const entityGeometry = new THREE.OctahedronGeometry( 0.05, 4 ); // 0.1, 4
const routerGeometry = new THREE.BoxGeometry(0.08, 0.08, 0.08); //0.08
const nodeMaterial = new THREE.MeshPhongMaterial({
    color:'#2CF604',
    emissive:'#000000',
    emissiveIntensity:1,
    specular:'#ffffff',
    shininess:30
});

const edgeConnectivityMaterial = new THREE.ShaderMaterial( {
    vertexShader: vertexShader,
    fragmentShader: fragmentShader,
    transparent: true,
} );

const connectivityMaterial = new THREE.LineBasicMaterial({
    color: '#ff2929'
});

const topologyMaterial = new THREE.LineBasicMaterial({
    color: '#fbff29'
});


// GUI
const effectController = {
    showConnectivity: true,
    showTopology: false,
    colorWithRisks: true,
    maxIter: 1950,
    stepSize: .015,
    alpha: 3.35,
    activateForce: true,
    Start: connectWebSocket,
    End: disconnectWebSocket
};

// Initialize
let namesArr = Array(maxNodes).fill('');
let topologyEdges=[], riskArr =  Array(maxNodes).fill(0), riskCov=Array(maxNodes).fill(Array(maxNodes).fill(0));
const allEdges= Array(maxNodes).fill(new Array(maxNodes).fill(0));
let nEntities;
const indDict = {}; // Dictionary of entity global indices{name:index}
let nNew;
let clusMemberships = [[]]; // Array of indices of entites in respective cluster
let activeNodes = [[]]; // indices of active nodes
let clusAssignments = {}; // Dictionary of entity cluster assignments {name:clusIndex}
let entityIndexInClus = {}; // Dictionary of 
let nodePosArr = [];

const diamXY = 2.3;
let diamZ = 2;
const bounds = {upper:[2.5, 2.5], lower:[-2.5, -2.5]};
let stepSize = effectController.stepSize;
let dt = stepSize / (effectController.maxIter + 1);
let alpha = effectController.alpha;

const nFrame = 2;
let counter = 0;

init();
//maxLabelEntity = labelMaxRisk(riskArr, maxLabelEntity, clusterGroup);
animate();
console.log(clusterGroup);

// Message Queue
function connectWebSocket(){
    ws = new WebSocket('ws://127.0.0.1:15674/ws');


    ws.onopen = function() {
        console.log('Connected to AMQP broker');

        const container = document.getElementById("container");
        container.className = 'slide';
    };


    ws.onmessage = function(event) {
        console.log('Received message from AMQP broker');
        let obj = JSON.parse(event.data);
        
        const funcEdges = obj.funcEdges;
        namesArr = obj.names;
        riskArr = obj.riskArr;
        riskCov = obj.riskCov;
        topologyEdges = obj.topologyEdges;
        nNew = obj.newEntities;
        //console.log(obj)

        nEntities = namesArr.length;
        let nFlows = obj.nFlows;
        let timeStamp = obj.timeStamp;
        const msg = `- ${timeStamp}: ${nFlows} flows`;

        // Add to HTML
        const para = document.createElement("p");
        para.classList.add('p1');
        const text = document.createTextNode(msg);
        para.appendChild(text);

        const logs = document.getElementById("logs");
        logs.appendChild(para);
        logs.scrollTop = logs.scrollHeight;

        // Update Nodes
        diamZ = Math.max(...riskArr);
        activateNodes(clusterGroup, clusMemberships, clusAssignments, entityIndexInClus, indDict, activeNodes, namesArr.length -  nNew, entityGeometry, namesArr, nodePosArr, riskArr, diamXY / 2, diamZ);
        //console.log(entityIndexInClus);
        updateNodeColors(riskArr, clusterGroup, nEntities, activeNodes);

        // Update Edges
        console.log('Before:',allEdges)
        updateEdgeArr(allEdges, funcEdges);
        console.log(allEdges)
        console.log(funcEdges)

        scene.remove(edgeConnectivity);
        edgeConnectivity = makeConnectivityEdges(edgeConnectivityMaterial, nodePosArr, allEdges);
        updateConnectivityColors(allEdges, edgeConnectivity, nEntities);
        scene.add(edgeConnectivity);
        edgeConnectivity.visible = effectController.showConnectivity;

        scene.remove(edgeTopology);
        edgeTopology = makeTopologyEdges(topologyMaterial, nodePosArr, topologyEdges, indDict);
        scene.add(edgeTopology);
        edgeTopology.visible = effectController.showTopology;
        
        //Label Some Entities
        //maxLabelEntity = labelMaxRisk(riskArr, maxLabelEntity, clusterGroup, clusAssignments, entityIndexInClus);
        console.log('Max Risk Entity: ', maxLabelEntity);

        // Reset Step Size
        stepSize = effectController.stepSize;

    };

    ws.onclose = function() {
        console.log('Disconnected from AMQP broker');
    };

}

function disconnectWebSocket() {
    ws.close();
}

function initGUI(){
    const gui = new GUI();

    const basic = gui.addFolder('Basics');

    basic.add( effectController, 'showConnectivity' ).onChange( function ( value ) {

        edgeConnectivity.visible = value

    } );

    basic.add( effectController, 'showTopology' ).onChange( function ( value ) {

        edgeTopology.visible = value

    } );

    basic.add( effectController, 'colorWithRisks' ).onChange( function ( value ) {
        
        
        for ( let j = 0; j < clusterGroup.children.length; j++ ) {

            const cluster = clusterGroup.children[j];
            

            for ( let i = 0; i < cluster.children.length; i++ ) {

                const node = cluster.children[i];
                const name = node.name;

                if (value == true) {
                    node.material.color.setRGB( risk_mean[name] / extras.diam_z , 0, 0);
                } else {
                    node.material.color.setRGB(entityColors[name][0], entityColors[name][1], entityColors[name][2]);
                }
            
            }
        }

        
    } );

    basic.add( effectController, 'maxIter', 50, 1000, 10).onChange( function ( value ){
        dt = stepSize / (value + 1);
    } );

    basic.add( effectController, 'stepSize', .001, .03, .001).onChange( function ( value ){
       stepSize=value;
    } );

    basic.add( effectController, 'alpha', .05, 5, .05).onChange( function ( value ){
        alpha=value;
     } );

    basic.add( effectController, 'activateForce' );

    //basic.close();

    const loadData = gui.addFolder('Load Data');

    loadData.add( effectController, 'Start' );

    loadData.add( effectController, 'End' );

}

function init(){ 
    
    initGUI();
    
    // Scene & Camera
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 4;
    scene.add(camera);

    // Legend
    [uiScene, orthoCamera] = generateLegend(fontPath, entityGeometry, routerGeometry, nodeMaterial, connectivityMaterial, topologyMaterial);
    
    // Lights
    scene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );
    //scene.background = new THREE.Color( 0xc4c4c4 );

    const light = new THREE.DirectionalLight( 0xffffff, 0.5 );
    light.position.set(1, 1, 1);
    scene.add( light );

    const uiLight = new THREE.DirectionalLight( 0xffffff, 0.5 );
    uiLight.position.set(1, 1, 1);
    uiScene.add( uiLight );

    uiScene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 8, 8 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: '#4a4a4a' } )
    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.position.z = -0.1;
    plane.receiveShadow = false;
    //scene.add( plane );

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.autoClear = false;
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    // Stats & Resize Window
    stats = new Stats();
    document.body.appendChild( stats.dom );

    window.addEventListener( 'resize', onWindowResize );

    const controls = new OrbitControls( camera, renderer.domElement );  

    // Geometries & Material
    
    // Nodes
    clusterGroup = new THREE.Group(); 
    clusterGroup.add(new THREE.Group() );
    addNodesSimple(clusterGroup, clusMemberships, clusAssignments, entityIndexInClus, 0, entityGeometry, namesArr, nodePosArr, riskArr, diamXY / 2, diamZ);
    scene.add( clusterGroup );
    for(const entity of clusterGroup.children[0].children){
        entity.visible = false;
    }

    // Edges

    // Connectivity
    edgeConnectivity = makeConnectivityEdges(edgeConnectivityMaterial, nodePosArr, allEdges);
    scene.add( edgeConnectivity );
    edgeConnectivity.visible = effectController.showConnectivity;

    // Topology 
    edgeTopology = makeTopologyEdges(topologyMaterial, nodePosArr, topologyEdges, indDict);
    scene.add( edgeTopology );
    edgeTopology.visible = effectController.showTopology;    
    
    // Cluster parameters
    //clusMemberships = [ []];
    //[clusMemberships, clusEdges] = computeClusterParams(clusterGroup, allEdges, clusAssignments, indDict);
    
}


function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    // Leave Legend Same Size
    orthoCamera.left = -2 * window.innerWidth / defWidth + 1;
    orthoCamera.top = 1 * window.innerHeight / defHeight;
    orthoCamera.bottom = -1 * window.innerHeight / defHeight;
    orthoCamera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

}

/**
 * Updates the all edges buffer with currently visible edges
 * @param {*} allEdges 
 * @param {*} funcEdges 
 */
function updateEdgeArr(allEdges, funcEdges){

    const nCur = funcEdges.length;

    for (let i=0; i < nCur; i++ ){
        for (let j=0; j < nCur; j++ ){
            allEdges[i][j] = funcEdges[i][j];
        }
    }
}

function updateNodeColors(riskArr, clusterGroup, nNodes, activeNodes){
    
    // Update Nodes
    const nodeColors = new Float32Array( nNodes * 4 );

    for (let i = 0, t, clr, normRisk; i < nNodes ; i++) {

        normRisk = riskArr[i] / diamZ;
        t = normRisk > 0 ? (normRisk <= 1 ? normRisk: 1): 0;
        
        clr = colormapLinear(color1, color2, t);

        nodeColors[ i * 4 ] = clr.r / 256;
        nodeColors[ i * 4 + 1] = clr.g / 256;
        nodeColors[ i * 4 + 2] =clr.b / 256;
        nodeColors[ i * 4 + 3] = 1;
    }

    for (let j=0; j < clusterGroup.children.length; j++) {
        for (let k=0, i, entity; k < activeNodes[j].length; k++){
            
            entity = clusterGroup.children[j].children[k];
            i = indDict[entity.name];
            entity.material.color.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2]);
        }
    }
}

function updateConnectivityColors(edgeWeightArr, edgeConnectivity, nNodes){

    const edgeColors = new Float32Array( 4 * 2 * nNodes * (nNodes - 1) );
    
    for (let i = 0; i < nNodes ; i++) {

        for (let j = 0; j < nNodes ; j++) {

            if (j == i){
                continue;
            }
            let k = i * nNodes + j;

            edgeColors[ 8 * k ] = (edgeWeightArr[i][j])** (1/3) * 255 ; 
            edgeColors[ 8 * k + 1] = 0;
            edgeColors[ 8 * k + 2] = 0;
            edgeColors[ 8 * k + 3] = (edgeWeightArr[i][j]) ** (3) * 255;

            edgeColors[ 8 * k + 4] = edgeColors[ 8 * k ]; 
            edgeColors[ 8 * k + 5] = edgeColors[ 8 * k + 1];
            edgeColors[ 8 * k + 6] = edgeColors[ 8 * k + 2];
            edgeColors[ 8 * k + 7] = edgeColors[ 8 * k + 3];
        }
    }
    edgeConnectivity.geometry.setAttribute( 'color', new THREE.Uint8BufferAttribute( edgeColors, 4, true ) );
}

//Find the min and max risk entities and add text label to them
function labelMaxRisk(riskArr, maxLabelEntity, clusterGroup, clusAssignments, entityIndexInClus){

    let indexOfMaxValue = riskArr.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
    //let indexOfMinValue = values.reduce((iMin, x, i, arr) => x < arr[iMin] ? i : iMin, 0);
    

    let name = namesArr[indexOfMaxValue];
    let entity = clusterGroup.children[ clusAssignments[name]].children[ entityIndexInClus[indexOfMaxValue]];

    
    if (maxLabelEntity != null && maxLabelEntity != name) {

        let oldMaxIndex = indDict[maxLabelEntity];
        let oldEntity = clusterGroup.children[ clusAssignments[maxLabelEntity]].children[ entityIndexInClus[oldMaxIndex]];
        oldEntity.remove(oldEntity.children[0]); // Remove old text
        
        // Add the text
        let size = 0.05;
        const fm = new fontManager(fontPath);
        const liteMat = new THREE.MeshBasicMaterial( {
            color: 0xffffff,
            transparent: true,
            opacity: .8,
            side: THREE.DoubleSide
        } );
        fm.addFont("Max Risk", [-size*2.5, -size/2., 0.05], liteMat, entity, size, [1, 1, 1]);
        const text = entity.children[0];
    }    

    return entity.name
}

//Mask the Array along axs given the boolean mask
function maskArray2(array, indices, axs=0) {
    
    const res = [];
    if (axs == 0) {
        indices.forEach( (i) => res.push(array[i]))
    } else if (axs == 1) {
        array.forEach( (row) => res.push( maskArray2(row, indices, 0) ) ) ;
    }
    return res;
}

// Move the nodes only within the cluster. TODO: Need to move to more robust datatype/table for sending data from py end
function moveNodes(clusterGroup, allPosArr, allEdgeWeights, clusMemberships, stepSize=null, diamXY=1.3, minDist = 0.001, alpha=1){
      
    const nClus = clusterGroup.children.length;
    // TODO: Right now moving according to all forces. Change to within cluster movement if you want
    allPosArr = calcMove(allPosArr, allEdgeWeights, stepSize, diamXY , bounds, minDist, alpha);
    

    for (let j = 0; j < nClus; j++){
        const cluster = clusterGroup.children[j];
        
        // Compute masked pos and weights for jth cluster
        const jClusIndices = clusMemberships[j];
        let clusPosArr =  tf.tidy( () => maskArray2(allPosArr, jClusIndices));
        const clusEdgeWeights =  tf.tidy( () => maskArray2( maskArray2(allEdgeWeights, jClusIndices, 0), jClusIndices, 1) );         

        setNodePos(cluster, clusPosArr);
        
    }
    
    return allPosArr;

}

// Move the nodes that are visible/active
function moveActiveNodes(clusterGroup, allPosArr, allEdgeWeights, clusMemberships, activeNodes, stepSize=null, diamXY=1.3, minDist = 0.001, alpha=1){
      
    const clusIndex = 0;
    
    // Compute masked pos and weights for active nodes in jth cluster
    const activeIndices = activeNodes[clusIndex];    
    let activePosArr =  tf.tidy( () => maskArray2(allPosArr, activeIndices));
    const activeEdgeWeights =  tf.tidy( () => maskArray2( maskArray2(allEdgeWeights, activeIndices, 0), activeIndices, 1) );         
    
    // TODO: Right now moving according to all forces. Change to within cluster movement if you want
    activePosArr = calcMove(activePosArr, activeEdgeWeights, stepSize, diamXY , bounds, minDist, alpha);
    
    // update allPosArr using activeIndices
    for(let i = 0; i < activeIndices.length; i++){
        allPosArr[activeIndices[i]] = activePosArr[i];
    }
    
    // Get the masked pos and weights for jth cluster
    const cluster = clusterGroup.children[clusIndex];
    const jClusIndices = clusMemberships[clusIndex];
    const clusPosArr =  tf.tidy( () => maskArray2(allPosArr, jClusIndices));
    
    setNodePos(cluster, clusPosArr);
    

}


function animate() {
    
    requestAnimationFrame( animate );

    render();

    stats.update();
}

function render() {
    const time = Date.now() * 0.001;

    if (effectController.activateForce){
        if ( counter % nFrame == 0 && activeNodes[clusIdx].length > 0 ) {
            //nodePosArr = moveNodes(clusterGroup, nodePosArr, allEdges, clusMemberships, stepSize, diamXY, .1, alpha); // 2.3
            
            moveActiveNodes(clusterGroup, nodePosArr, allEdges, clusMemberships, activeNodes, stepSize, diamXY, .1, alpha); // 2.3
            stepSize = (stepSize > dt) ? stepSize - dt : 0;
            
            setAllEdgePosFromNodePos(edgeConnectivity, nodePosArr);
            setEdgePosFromNodePos(edgeTopology, nodePosArr, topologyEdges, indDict);
        }
        counter += 1;
    }
    
    renderer.clear();
	renderer.render( scene, camera );
    renderer.render( uiScene, orthoCamera );
}



