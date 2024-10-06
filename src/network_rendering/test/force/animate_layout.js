import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import vertexShader from '../../shaders/vertex.glsl.js'
import fragmentShader from '../../shaders/fragment.glsl.js'

import {generateLegend} from '../legend/legendMaker.js';
import {makeNodes, makeConnectivityEdges, setNodePos, setEdgePosFromNodePos, computeClusterParams} from '../hierarchical/graphMaker.js';

import {calcMove} from '../force/force-directed.js'
import * as data from '../../saves/net_data_medium3.json' assert {type: 'json'}; // 63
console.log(data);

const fontPath = 'fonts/helvetiker_regular.typeface.json';

let camera, scene, renderer, stats;
let clusterGroup, clusMemberships, clusEdges;
let edgeConnectivity, edgePos;
let uiScene, orthoCamera;

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

const topologyMaterial = new THREE.LineBasicMaterial({
    color: '#fbff29'
});


// GUI
const effectController = {
    showConnectivity: true,
    colorWithRisks: true,
    maxIter: 1950,
    stepSize: .015,
    alpha: 3.35,
    activateForce: false
};

// Read planar positions
const {pos, topologyEdges, risk_mean, risk_cov, funcEdges, entityColors, clusAssignments, extras} = data
const namesArr = Object.keys(pos);
const nNodes = namesArr.length;
const indDict = {}; // Dictionary of {name:index}
for (let i = 0; i < nNodes; i++) {
    indDict[namesArr[i]] = i;
}

let allNodePos = Array.from(Object.keys(pos), (key) => pos[key]);
let stepSize = effectController.stepSize;
let dt = stepSize / (effectController.maxIter + 1);
let alpha = effectController.alpha
const bounds = {upper:[2.5, 2.5], lower:[-2.5, -2.5]};

const nFrame = 2;
let counter = 0;

init();
animate();

function initGUI(){
    const gui = new GUI();

    gui.add( effectController, 'showConnectivity' ).onChange( function ( value ) {

        edgeConnectivity.visible = value

    } );

    gui.add( effectController, 'colorWithRisks' ).onChange( function ( value ) {
        
        
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

    gui.add( effectController, 'maxIter', 50, 1000, 10).onChange( function ( value ){
        dt = stepSize / (value + 1);
    } );

    gui.add( effectController, 'stepSize', .001, .03, .001).onChange( function ( value ){
       stepSize=value;
    } );

    gui.add( effectController, 'alpha', .05, 5, .05).onChange( function ( value ){
        alpha=value;
     } );

    gui.add( effectController, 'activateForce' );

    //gui.close();
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
    scene.add( plane );

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

    clusterGroup = makeNodes(entityGeometry, routerGeometry, pos, funcEdges, risk_mean, entityColors,
        clusAssignments, extras, sizeMult, effectController.colorWithRisks); // Entity nodes and edges
    scene.add( clusterGroup );

    // Edges

    // Connectivity
    edgeConnectivity = makeConnectivityEdges(edgeConnectivityMaterial, pos, funcEdges, risk_mean);
    
    scene.add( edgeConnectivity );
    edgeConnectivity.visible = effectController.showConnectivity;
    
    // Cluster parameters
    [clusMemberships, clusEdges] = computeClusterParams(clusterGroup, funcEdges, clusAssignments, indDict);
    
    
    
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
    
    const time = Date.now() * 0.001;

    if (effectController.activateForce){
        if ( counter % nFrame == 0) {
            allNodePos = moveNodes(clusterGroup, allNodePos, funcEdges, clusMemberships, stepSize, 1.3, .1, alpha); // 2.3
            stepSize = (stepSize > dt) ? stepSize - dt : 0;
            
            setEdgePosFromNodePos(edgeConnectivity, allNodePos);
        }
        counter += 1;
    }
    
    renderer.clear();
	renderer.render( scene, camera );
    renderer.render( uiScene, orthoCamera );

    requestAnimationFrame( animate );

    stats.update();
}




