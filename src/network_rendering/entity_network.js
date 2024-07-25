import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';
import TWEEN from '@tweenjs/tween.js'

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';

import vertexShader from '/shaders/vertex.glsl.js'
import fragmentShader from '/shaders/fragment.glsl.js'
import vertexShaderDefault from '/shaders/vertex_default.glsl.js'
import fragmentShaderDefault from '/shaders/fragment_default.glsl.js'

import {generateLegend, fontManager} from '/test/legend/legendMaker.js';
import {makeNodes, makeConnectivityEdges, makeTopologyEdges, setNodePos, setAllEdgePosFromNodePos, setEdgePosFromNodePos,
    computeClusterParams, colormapLinear, color1, color2} from '/test/hierarchical/graphMaker.js';

import {calcMove} from '/test/force/force-directed.js'
import * as data from '/saves/net_data_medium0.json' assert {type: 'json'}; // medium1

console.log(data);

const fontPath = '/fonts/helvetiker_regular.typeface.json';

let camera, scene, renderer, stats, greeter;
let clusterGroup, clusMemberships, clusEdges, entityIndexInClus;
let edgeConnectivity, edgeTopology;
let uiScene, orthoCamera;
let ws;
let maxLabelEntity = null;

// Legend Parameters
const defWidth = 900; 
const defHeight = 500; 

// Node & Edge Parameters
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

let topologyMaterial = new THREE.LineBasicMaterial({
    color: '#fbff29',
    linewidth: 0.5
});

topologyMaterial = new THREE.ShaderMaterial({
    vertexShader: vertexShaderDefault,
    fragmentShader: fragmentShaderDefault,
    transparent: true,
});


// GUI
const effectController = {
    showConnectivity: false,
    showTopology: true,
    colorWithRisks: true,
    maxIter: 1950,
    stepSize: .015,
    alpha: 3.35,
    activateForce: true,
    Start: connectWebSocket,
    End: disconnectWebSocket
};

// Read planar positions
let {namesArr, nodePosArr, topologyEdges, riskArr, riskCov: riskCov, funcEdges, entityColors, clusAssignments, extras} = data
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


init();
maxLabelEntity = labelMaxRisk(riskArr, maxLabelEntity, clusterGroup);
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
        
        funcEdges = obj.funcEdges;
        riskArr = obj.riskArr;
        riskCov = obj.riskCov;
        topologyEdges = obj.topologyEdges;



        const streamNames = obj.names;
        let nFlows = obj.nFlows;
        let timeStamp = obj.timeStamp;
        const msg = `- ${timeStamp}: ${nFlows} flows`;

        //console.log(funcEdges);
        //console.log(topologyEdges);
        //console.log(riskArr);

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
        updateNodeColors(riskArr, clusterGroup, nNodes)
        
        //Label Some Entities
        maxLabelEntity = labelMaxRisk(riskArr, maxLabelEntity, clusterGroup);
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

    basic.close();

    const loadData = gui.addFolder('Load Data');

    loadData.add( effectController, 'Start' );

    loadData.add( effectController, 'End' );

}

function init(){ 
    
    // Scene & Camera
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    //camera.position.z = 4;
    camera.position.set(0, 0, 4);
    scene.add(camera);

    uiScene = new THREE.Scene();
    orthoCamera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, .1, 2 );
   
    // Lights
    scene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );
    //scene.background = new THREE.Color( 0xc4c4c4 );

    const light = new THREE.DirectionalLight( 0xffffff, 0.5 );
    light.position.set(1, 1, 1);
    scene.add( light );

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 8, 8 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: '#4a4a4a' } )
    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.position.z = -0.1;
    plane.receiveShadow = false;
    scene.add( plane );

    // Greeter
    const loader = new FontLoader();    
    const fontSize = 5;

    loader.load( fontPath, function ( font ) {

        const liteMat = new THREE.MeshBasicMaterial( {
            color: 0xffffff,
            transparent: true,
            opacity: .8,
            side: THREE.DoubleSide
        } );
    
        const geometryTitle = new TextGeometry( 'NRE', { 
            font: font,    
            size: fontSize,
            curveSegments: 3,
            bevelEnabled: false,
        }  );
        geometryTitle.scale(1, 1, .005);  
        geometryTitle.center();
        
        const geometrySubTitle = new TextGeometry( '- Network Risk Estimation-\n \xa0\xa0 Visualization Tool', { 
            font: font,    
            size: fontSize/3,
            curveSegments: 3,
            bevelEnabled: false,
        }  );
        geometrySubTitle.scale(1, 1, .005); 
        geometrySubTitle.center();
        greeter = new THREE.Mesh( geometryTitle, liteMat );
        
        const textSubTitle = new THREE.Mesh( geometrySubTitle, liteMat );
        textSubTitle.translateY( -fontSize * 1.4)
        greeter.add(textSubTitle);

        const geometryDesc = new TextGeometry( 'Click Anywhere', { 
            font: font,    
            size: fontSize/8,
            curveSegments: 3,
            bevelEnabled: false,
        }  );
        geometryDesc.scale(1, 1, .002); 
        geometryDesc.center();
        
        const textDesc = new THREE.Mesh( geometryDesc, liteMat );
        textDesc.translateY( -fontSize * 2.1)
        greeter.add(textDesc);

        greeter.rotateX( Math.PI / 2);
        greeter.position.set(0, 7, 20);
        scene.add(greeter);

        camera.position.set(0, -15, 20);
        camera.lookAt(greeter.position);
        camera.position.set(0, -15, 15);
    } );

    document.addEventListener('click', panToPlane);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.autoClear = false;
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    // Stats & Resize Window
    stats = new Stats();
    document.body.appendChild( stats.dom );

    window.addEventListener( 'resize', onWindowResize );

    //const controls = new OrbitControls( camera, renderer.domElement );  

    // Geometries & Material
    
    // Nodes
    [clusterGroup, entityIndexInClus] = makeNodes(entityGeometry, routerGeometry, namesArr,  nodePosArr, funcEdges, riskArr, entityColors,
        clusAssignments, extras, sizeMult, effectController.colorWithRisks); // Entity nodes and edges
    scene.add( clusterGroup );
    //console.log( entityIndexInClus)

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
    
}

// Pans the camera to the initial plane
function panToPlane() {
    // Create a tweens
    const myEasing = TWEEN.Easing.Sinusoidal.In;
    
    new TWEEN.Tween(camera.position).to({
        x: 0,
        y: 0,
        z: 4
    }, 1000).easing(myEasing).start();
    
    new TWEEN.Tween(camera.rotation).to({
        x: 0,
    }, 1000).easing(myEasing).start();     

    // Legend
    [uiScene, orthoCamera] = generateLegend(fontPath, entityGeometry, routerGeometry, nodeMaterial, connectivityMaterial, topologyMaterial);
    const uiLight = new THREE.DirectionalLight( 0xffffff, 0.5 );
    uiLight.position.set(1, 1, 1);
    uiScene.add( uiLight );
    uiScene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );

    // GUI
    initGUI();

    // Controls
    const controls = new OrbitControls( camera, renderer.domElement );  

    document.removeEventListener('click', panToPlane);
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

function updateNodeColors(riskArr, clusterGroup, nNodes){
    
    // Update Nodes
    const nodeColors = new Float32Array( nNodes * 4 );

    for (let i = 0, t, clr, normRisk; i < nNodes ; i++) {

        normRisk = riskArr[i] / extras.diam_z
        t = normRisk > 0 ? (normRisk <= 1 ? normRisk: 1): 0;
        
        clr = colormapLinear(color1, color2, t);

        nodeColors[ i * 4 ] = clr.r / 256;
        nodeColors[ i * 4 + 1] = clr.g / 256;
        nodeColors[ i * 4 + 2] =clr.b / 256;
        nodeColors[ i * 4 + 3] = 1;
    }

    for (let j=0; j < clusterGroup.children.length; j++) {
        for (let k=0, i, entity; k < clusterGroup.children[j].children.length; k++){
            
            entity = clusterGroup.children[j].children[k];
            i = indDict[entity.name];
            entity.material.color.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2]);
        }
    }
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
function labelMaxRisk(riskArr, maxLabelEntity, clusterGroup){

    let indexOfMaxValue = riskArr.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
    //let indexOfMinValue = values.reduce((iMin, x, i, arr) => x < arr[iMin] ? i : iMin, 0);
    

    let name = namesArr[indexOfMaxValue];
    let entity = clusterGroup.children[ clusAssignments[name]].children[ entityIndexInClus[indexOfMaxValue]];

    
    if (maxLabelEntity != null && maxLabelEntity != name) {

        let oldMaxIndex = indDict[maxLabelEntity];
        let oldEntity = clusterGroup.children[ clusAssignments[maxLabelEntity]].children[ entityIndexInClus[oldMaxIndex]];
        oldEntity.remove(oldEntity.children[0]);
        
        // Add the text
        let size = 0.1; // 0.05
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

    stats.update();
}

function render() {
    const time = Date.now() * 0.001;

    if (effectController.activateForce){
        if ( counter % nFrame == 0) {
            nodePosArr = moveNodes(clusterGroup, nodePosArr, funcEdges, clusMemberships, stepSize, 1.3, .1, alpha); // 2.3
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



