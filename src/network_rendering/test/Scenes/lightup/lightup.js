// This script is simulates two scenes which convey the information provided by NRE
// 3/20/2025

import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';
import TWEEN from '@tweenjs/tween.js'

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';

import vertexShader from '../../../shaders/vertex.glsl.js'
import fragmentShader from '../../../shaders/fragment.glsl.js'
import vertexShaderDefault from '../../../shaders/vertex_default.glsl.js'
import fragmentShaderDefault from '../../../shaders/fragment_default.glsl.js'

import {generateLegend, fontManager} from '../../../test/legend/legendMaker.js';
import {makeNodes, makeConnectivityEdges, makeTopologyEdges, setNodePos, setAllEdgePosFromNodePos, setEdgePosFromNodePos,
    computeClusterParams, colormapLinear, color1, color2} from '../../../test/hierarchical/graphMaker.js';

import {calcMove} from '../../../test/force/force-directed.js'
import * as data from '../../../saves/net_data_medium1.json' assert {type: 'json'}; // medium1

import jsonAll from '../../stream_data/render_data_medium1.json'
import {addEntityBloomClustered,  nonBloomed, restoreMaterial} from '../../prettify/postProcess.js'

const jsonObjAll = JSON.parse(jsonAll);

const fontPath = '/fonts/helvetiker_regular.typeface.json';
const streamDataPath = '../stream_data';

let camera, scene, renderer, stats, greeter;
let clusterGroup, clusMemberships, clusEdges, entityIndexInClus;
let edgeConnectivity, edgeTopology;
let uiScene, orthoCamera;
let maxLabelEntityName = null;
let show = false;

// Whether recording mode or not
const recordMode = true;
const nreOn = false;

let bloomComposer, finalComposer, bloomPass, bloomLayer;
const nodeMaterials = [];

// Legend Parameters
const defWidth = 900; 
const defHeight = 500; 

// Node & Edge Parameters
const sizeMult = .5;
const entityGeometry = new THREE.OctahedronGeometry( 0.05, 4 ); // 0.1, 4
const routerGeometry = new THREE.BoxGeometry(0.08, 0.08, 0.08); //0.08
const entitySampleMaterial = new THREE.MeshPhongMaterial({
    color:'#000000',
    emissive:'#000000',
    emissiveIntensity: 3,
    specular:'#ffffff',
    shininess:30
});

const edgeConnectivityMaterial = new THREE.ShaderMaterial( {
    vertexShader: vertexShader,
    fragmentShader: fragmentShader,
    transparent: true,
} );

const edgeConnectivityMaterial2 = new THREE.LineBasicMaterial({
    color: '#ff2929'
});

const topologyMaterial = new THREE.LineBasicMaterial({
    color: '#fbff29',
    linewidth: 0.5
});

/*
const topologyMaterial = new THREE.ShaderMaterial({
    vertexShader: vertexShaderDefault,
    fragmentShader: fragmentShaderDefault,
    transparent: true,
});
*/

// GUI
const effectController = {
    showConnectivity: false,
    showTopology: true,
    colorWithRisks: true,
    maxIter: 1950,
    stepSize: .015,
    alpha: 3.35,
    activateForce: true,
    Start: startStream,
    End: endStream
};

// Post Processing Parameters
const ppParams = {
    threshold: -0.6, // -0.6
    strength: 0.147, // 0.515
    radius: 0.37, // 0.37
    exposure: 1.36, // 1.0
    innerWidth : window.innerWidth,
    innerHeight : window.innerHeight,
    emmisiveIntensity: entitySampleMaterial.emissiveIntensity
};

// Read planar positions
let {namesArr, nodePosArr, topologyEdges, riskArr, risk_cov: riskCov, funcEdges, entityColors, clusAssignments, extras} = data
const nNodes = namesArr.length;
const indDict = {}; // Dictionary of {name:index}
for (let i = 0; i < nNodes; i++) {
    indDict[namesArr[i]] = i;
}

let stepSize = effectController.stepSize;
let dt = stepSize / (effectController.maxIter + 1);
let alpha = effectController.alpha
const bounds = {upper:[2.5, 2.5], lower:[-2.5, -2.5]};

const nFrame = 2;
let counter = 0;
let stopVar = false;

const startTime = Date.now();
const slantDeg = 10 /180 * Math.PI; // Tilt of the plane of camera path
const initPos = new THREE.Vector3( 0, -2.8, 1.4 ); // -0.0065, -3.886, 0.948
const cRadius = Math.sqrt(initPos.x ** 2 + initPos.y **2);

init();
// maxLabelEntityName = labelMaxRisk(riskArr, maxLabelEntityName, clusterGroup, 'perp');

animate();
console.log(clusterGroup);

// Read from json files
async function advanceStreamTick(ind){
    //const fileName = streamDataPath + '/render_data_' + ind.toString().padStart(3, '0') + '.json';
    //console.log(fileName);
    //const response = await fetch(fileName);
    
    //const json = await response.json(); // This one throws error
    //console.log(json);
    const jsonObj = jsonObjAll[ind];
    

    // Use the json data
    processStreamTick(jsonObj);
}

function processStreamTick(jsonObj) {
    funcEdges = jsonObj.funcEdges;
    riskArr = jsonObj.riskArr;
    riskCov = jsonObj.riskCov;
    topologyEdges = jsonObj.topologyEdges;

    const streamNames = jsonObj.names;
    let nFlows = jsonObj.nFlows;
    let timeStamp = jsonObj.timeStamp;
    const msg = `- ${timeStamp}: ${nFlows} flows`;

    // Add to HTML
    const para = document.createElement("p");
    para.classList.add('p1');
    const text = document.createTextNode(msg);
    para.appendChild(text);

    const logs = document.getElementById("logs");
    logs.appendChild(para);
    logs.scrollTop = logs.scrollHeight;

    // Update Edges
    updateConnectivityColors(funcEdges, edgeConnectivity, nNodes);

    scene.remove(edgeTopology);
    edgeTopology = makeTopologyEdges(topologyMaterial, nodePosArr, topologyEdges, indDict);
    scene.add(edgeTopology);
    edgeTopology.visible = effectController.showTopology;
    //setEdgePosFromNodePos(edgeTopology, allNodePos, topologyEdges, indDict);

    // Update Nodes
    //updateNodeColors(riskArr, clusterGroup, nNodes);
    updateNodeColors(riskArr, riskCov, clusterGroup, entityIndexInClus, ppParams, nreOn);
    
    //Label Some Entities
    if (nreOn){
        maxLabelEntityName = labelMaxRisk(riskArr, maxLabelEntityName, clusterGroup);//, 'perp');
        console.log('Max Risk Entity: ', maxLabelEntityName);  
    }
    

    // Reset Step Size
    stepSize = effectController.stepSize;

}

//let myPromise;
async function streamLoop(delay) {
    let ind = 0;
    while (ind <= 100 && !stopVar) {
        try {
            let myPromise = await new Promise(resolve => setTimeout(() => {resolve(advanceStreamTick(ind))}, delay));
            ind += 1;
            //console.log(ind);
        } catch (error) {
            console.error(error.message);
        }
    }
}


async function startStream() {

    console.log('Starting the feed');

    const container = document.getElementById("container");
    //container.className = 'slide';

    try {

        await streamLoop(2500);      
        
    } catch (error) {
        console.error(error.message);
    }
}

// TODO: Stop immediately
function endStream() {
    //myPromise.resolve();
    stopVar = true;
    setTimeout(() => {stopVar = false}, 3500);
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

    basic.close();

    const postProcessingFolder = gui.addFolder( 'Post Processing' );
    postProcessingFolder.add( ppParams, 'threshold', -1.0, 1.0 ).onChange( function ( value ) {
        bloomPass.threshold = Number( value );
    } );
    postProcessingFolder.add( ppParams, 'strength', 0.0, 3.0 ).onChange( function ( value ) {  
        bloomPass.strength = Number( value );
    } );
    postProcessingFolder.add( ppParams, 'radius', 0.0, 1.0 ).step( 0.01 ).onChange( function ( value ) {
        bloomPass.radius = Number( value );
    } );
    postProcessingFolder.add( ppParams, 'exposure', 0.1, 2 ).onChange( function ( value ) {
        renderer.toneMappingExposure = Math.pow( value, 3.0 );
    } );
    postProcessingFolder.close();

    const loadData = gui.addFolder('Load Data');

    loadData.add( effectController, 'Start' );

    loadData.add( effectController, 'End' );

}

function init(){ 
    
    // Scene & Camera
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    //camera.position.z = 4;
    camera.position.set( initPos.x, initPos.y, initPos.z) ; // 0, 0, 4
    camera.lookAt(0, 0, 0);
    scene.add(camera);

    uiScene = new THREE.Scene();
    orthoCamera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, .1, 2 );
   
    // Lights
    scene.add( new THREE.AmbientLight( 0xf0f0f0, 0.2 ) ); //0.1
    //scene.background = new THREE.Color( 0xc4c4c4 );

    const light = new THREE.DirectionalLight( 0xffffff, 0.4 ); // 0.4
    light.position.set(1, 1, 1);
    scene.add( light );

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 8, 8 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: '#4a4a4a' } )
    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.position.z = -0.1;
    plane.receiveShadow = false;
    scene.add( plane );

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.autoClear = false;
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    renderer.toneMapping = THREE.CineonToneMapping;
    renderer.toneMappingExposure = Math.pow( ppParams.exposure, 3.0 ); // 4.0
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // GUI
    if (!recordMode){
        initGUI();
    }    

    // Controls
    // const controls = new OrbitControls( camera, renderer.domElement );  

    // Stats & Resize Window
    if (!recordMode){   
        stats = new Stats();
        document.body.appendChild( stats.dom );
    }
    window.addEventListener( 'resize', onWindowResize );
    
    // Nodes
    if (!nreOn) {
        // Make Grid instead of custom layout
        const maxOfEachRow = nodePosArr.map(row => Math.max(...row));
        const nGrid = Math.floor(Math.sqrt(namesArr.length));
        const deltaXY = Math.max(...maxOfEachRow) / nGrid * 2.5;
        //console.log(deltaXY);

        namesArr.forEach( (name, i) => {            
            let gridX = ((i % nGrid) - nGrid/2) * deltaXY;
            let gridY = (Math.floor(i / nGrid) - nGrid/2) * deltaXY;
            nodePosArr[i] = [gridX, gridY];
        });
    }
    [clusterGroup, entityIndexInClus] = makeNodes(entityGeometry, routerGeometry, namesArr,  nodePosArr, funcEdges, riskArr, entityColors,
        clusAssignments, extras, sizeMult, effectController.colorWithRisks, entitySampleMaterial, nodeMaterials, nreOn); // Entity nodes and edges
    scene.add( clusterGroup );
    //console.log( entityIndexInClus) // -> Withing cluster the index of an entity

    // Edges
    // Connectivity
    edgeConnectivity = makeConnectivityEdges(edgeConnectivityMaterial, nodePosArr, funcEdges);
    
    scene.add( edgeConnectivity );
    edgeConnectivity.visible = effectController.showConnectivity;

    // Topology 
    edgeTopology = makeTopologyEdges(topologyMaterial, nodePosArr, topologyEdges, indDict);
    scene.add( edgeTopology );
    edgeTopology.visible = effectController.showTopology;
    
    // Cluster parameters
    [clusMemberships, clusEdges] = computeClusterParams(clusterGroup, funcEdges, clusAssignments, indDict);
    
    // Label max entity
    if (nreOn) {
        maxLabelEntityName = labelMaxRisk(riskArr, maxLabelEntityName, clusterGroup);//, 'perp');
    }
    //console.log(maxLabelEntityName);
    
    // Posprocessing Passes
    console.log(clusterGroup);
    [bloomComposer, finalComposer, bloomPass, bloomLayer ] = addEntityBloomClustered(riskCov, clusterGroup, clusAssignments, entityIndexInClus, namesArr,
        scene, camera, renderer, ppParams);
    //console.log(bloomLayer.idx);
    //edgeConnectivity.layers.enable(bloomLayer.idx);
    
    if (recordMode) {
        edgeConnectivity.visible = false;
        edgeTopology.visible = false;
        startStream();
        //edgeTopology.visible = true;
    }
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
    bloomComposer.setSize( window.innerWidth, window.innerHeight );
    finalComposer.setSize( window.innerWidth, window.innerHeight );
}

function updateNodeColors(riskArr, riskCov, clusterGroup, entityIndexInClus, bloomParams, nreOn=true){
    
    // Update Nodes
    const nNodes = entityIndexInClus.length;
    const nodeColors = new Float32Array( nNodes * 4 );

    for (let i = 0, t, clr, normRisk; i < nNodes ; i++) {

        normRisk = riskArr[i] / extras.diam_z
        t = normRisk > 0 ? (normRisk <= 1 ? normRisk: 1): 0;
        
        clr = colormapLinear(color1, color2, t);

        nodeColors[ i * 4 ] = clr.r / 256;
        nodeColors[ i * 4 + 1] = clr.g / 256;
        nodeColors[ i * 4 + 2] = clr.b / 256;
        nodeColors[ i * 4 + 3] = 1;
    }

    for (let j=0; j < clusterGroup.children.length; j++) {
        for (let k=0, i, entity; k < clusterGroup.children[j].children.length; k++){
            
            entity = clusterGroup.children[j].children[k];
            i = indDict[entity.name];
            entity.material.color.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2]);
            if (nreOn) {
                entity.material.color.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2]);
                entity.material.emissive.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2] );
            } else {
                entity.material.color.setRGB(0, 0, 0);
                entity.material.emissive.setRGB(0, 0, 0 );
            }
        }
    }

    // Update emissive mults
    const nodeVar = Array.from(riskCov, (elem, i) => elem[i]);
    const maxVar = Math.max(...nodeVar);
    const minVar = Math.min(...nodeVar);
    //console.log(nodeVar.slice(0, 5));
    //console.log(maxVar / minVar);

    entityIndexInClus.forEach( (ind, i) => {
        let name = namesArr[i];
        let node = clusterGroup.children[ clusAssignments[ name]].children[ind];
        
        node.material.emissiveMult = Math.pow( minVar / nodeVar[i], 3);
        node.material.emissiveIntensity = bloomParams.emmisiveIntensity * node.material.emissiveMult;
        
    });
}

function updateConnectivityColors(funcEdges, edgeConnectivity, nNodes){

    const edgeColors = new Float32Array( 4 * 2 * nNodes * (nNodes - 1) );
    
    for (let i = 0; i < nNodes ; i++) {

        for (let j = 0; j < nNodes ; j++) {

            if (j == i){
                continue;
            }
            let k = i * nNodes + j;

            edgeColors[ 8 * k ] = (funcEdges[i][j])** (1/3) * 255 ; 
            edgeColors[ 8 * k + 1] = 0;
            edgeColors[ 8 * k + 2] = 0;
            edgeColors[ 8 * k + 3] = (funcEdges[i][j]) ** (3) * 255;

            edgeColors[ 8 * k + 4] = edgeColors[ 8 * k ]; 
            edgeColors[ 8 * k + 5] = edgeColors[ 8 * k + 1];
            edgeColors[ 8 * k + 6] = edgeColors[ 8 * k + 2];
            edgeColors[ 8 * k + 7] = edgeColors[ 8 * k + 3];
        }
    }
    edgeConnectivity.geometry.setAttribute( 'color', new THREE.Uint8BufferAttribute( edgeColors, 4, true ) );
}

//Find the min and max risk entities and add text label to them
function labelMaxRisk(riskArr, maxLabelEntityName, clusterGroup){

    let textRotation;
    let indexOfMaxValue = riskArr.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
    //let indexOfMinValue = values.reduce((iMin, x, i, arr) => x < arr[iMin] ? i : iMin, 0);
    

    let name = namesArr[indexOfMaxValue];
    // Cluster -> Entity -> Text
    let entity = clusterGroup.children[ clusAssignments[name]].children[ entityIndexInClus[indexOfMaxValue]];

    
    if (maxLabelEntityName == null || maxLabelEntityName != name) { // If label is not on maxRiskEntity
        
        if (maxLabelEntityName != null) {
            let oldMaxIndex = indDict[maxLabelEntityName];
            let oldEntity = clusterGroup.children[ clusAssignments[maxLabelEntityName]].children[ entityIndexInClus[oldMaxIndex]];
            
            // Remove the old label which is a children of entity
            oldEntity.remove(oldEntity.children[0]);
        }
        
        
        // Add the text
        let size = 0.1; // 0.05
        const fm = new fontManager(fontPath);
        const liteMat = new THREE.MeshBasicMaterial( {
            color: 0xffffff,
            transparent: true,
            opacity: 1.,
            side: THREE.DoubleSide
        } );
        
        // Add new text as children to the entity
        // Async
        // For textpos the default orientation in facing in z direction where text is in xy plane
        // The [x, y, z] position corresponds to scene's [x, y, z] before text object rotation
        fm.addFont("Max Risk", [-size*2.5, 0.05, 0], liteMat, entity, size, [1, 1, 1], textRotation); // z 0.05
        //let text = entity.children[0]; // Doesnt work due to asynch runtime
        
    }    

    return entity.name
}

// Given the name returns the object related to that entity
function  maxRiskFaceCamera(maxLabelEntityName){

    const ind = namesArr.indexOf(maxLabelEntityName);
    const entity = clusterGroup.children[ clusAssignments[maxLabelEntityName]].children[ entityIndexInClus[ind]];

    const textObj = entity.children[0];
    
    textObj.lookAt(camera.position);
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


function animate() {
    
    requestAnimationFrame( animate );

    TWEEN.update();

    render();
    if (!recordMode){
        stats.update();
    }
    
}

function render() {

    
    const time = (Date.now() - startTime) * 0.0005;
    camera.position.set( cRadius * Math.sin(time), -cRadius * Math.cos(time),
     cRadius * Math.cos(time) * Math.tan(slantDeg ) + initPos.z )
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, 0);
    if (nreOn){
        maxRiskFaceCamera(maxLabelEntityName);
    }
    //let posVector = new THREE.Vector3();
    

    if (effectController.activateForce){
        if ( counter % nFrame == 0 && nreOn) {
            nodePosArr = moveNodes(clusterGroup, nodePosArr, funcEdges, clusMemberships, stepSize, 1.3, .1, alpha); // 2.3
            stepSize = (stepSize > dt) ? stepSize - dt : 0;
            
            setAllEdgePosFromNodePos(edgeConnectivity, nodePosArr);
            setEdgePosFromNodePos(edgeTopology, nodePosArr, topologyEdges, indDict);
        }
        counter += 1;
    }
    
    // Do every 2 seconds
    if (Math.floor(time) % 2 == 0 ){
        if (show){
            //camera.getWorldPosition(posVector);
            //console.log(posVector);
            //maxRiskFaceCamera(maxLabelEntityName);
            show = false;
        }         
    } else {
        show = true;
    }
    
    
    renderer.clear();
    if (!nreOn){
        renderer.render( scene, camera );
    } else {
        scene.traverse((obj) => {nonBloomed(obj, bloomLayer)});
        bloomComposer.render();  
        scene.traverse(restoreMaterial);
        finalComposer.render();
    }     
    renderer.render( uiScene, orthoCamera );

}



