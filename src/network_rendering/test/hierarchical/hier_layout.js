import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import vertexShader from '../../shaders/vertex.glsl.js'
import fragmentShader from '../../shaders/fragment.glsl.js'

import {generateLegend} from '../legend/legendMaker.js';
import {makeNodes, makeConnectivityEdges} from './graphMaker.js';
import {singleStepForceDirected, scaleToBounds} from '../force/force-directed.js'
import * as data from '../../saves/net_data_medium2.json' assert {type: 'json'}; // 63

console.log(data);

const fontPath = 'fonts/helvetiker_regular.typeface.json';

let camera, scene, renderer, stats;
let entityGroup;
let nodeColors, nodePosArray, nodeSizes;
let edgeConnectivity, edgeColors, edgePos;
let uiScene, orthoCamera;

// Legend Parameters
const defWidth = 900; 
const defHeight = 500; 

// Node & Edge Parameters
const sizeMult = .5;
const entityGeometry = new THREE.OctahedronGeometry( 0.05, 4 ); // 0.1, 4
const routerGeometry = new THREE.OctahedronGeometry( 0.055, 0 ); // 0.11, 0
const nodeMaterial = new THREE.MeshPhongMaterial({
    color:'#000000',
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
    maxIter: 50,
    stepSize: .1,
    activateForce: false
};

// Read planar positions
const {pos, topologyEdges, risk_mean, risk_cov, funcEdges, entityColors, extras} = data
//let {pos, risk, edges, entityColors, extras} = generateSampleNet(0, 0 ,2);
const nNodes = Object.keys(pos).length;

let nodePos = Array.from(Object.keys(pos), (key) => pos[key]); // Array of node positions sorted acc. to pos
let stepSize = .01;
let dt = stepSize / (effectController.maxIter + 1);

const nFrame = 5;
let counter = 0;

init();
animate();

function initGUI(){
    const gui = new GUI();

    gui.add( effectController, 'showConnectivity' ).onChange( function ( value ) {

        edgeConnectivity.visible = value

    } );

    gui.add( effectController, 'colorWithRisks' ).onChange( function ( value ) {

        if (value == true) {
            for ( let i = 0; i < nNodes; i ++ ) {

                let name = Object.keys(pos)[i];
                let node = entityGroup.children[i];

                node.material.color.setRGB( risk_mean[name] / extras.diam_z , 0, 0);
            
            }
        } else {
            for ( let i = 0; i < nNodes; i ++ ) {
                let name = Object.keys(pos)[i];
                let node = entityGroup.children[i];
                node.material.color.setRGB(entityColors[name][0], entityColors[name][1], entityColors[name][2]);
        
            }
        }
    } );

    gui.add( effectController, 'maxIter', 10, 100, 10).onChange( function ( value ){
        dt = stepSize / (value + 1);
    } );

    gui.add( effectController, 'stepSize', .05, .5, .05).onChange( function ( value ){
       stepSize=value;
    } );

    gui.add( effectController, 'activateForce' );
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
    scene.background = new THREE.Color( 0xc4c4c4 );
    scene.add( new THREE.AmbientLight( 0xf0f0f0, 0.8 ) );    

    const light = new THREE.DirectionalLight( 0xffffff, 0.5 );
    light.position.set(1, 1, 1);
    scene.add( light );

    const uiLight = new THREE.DirectionalLight( 0xffffff, 0.5 );
    uiLight.position.set(1, 1, 1);
    uiScene.add( uiLight );

    uiScene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 8, 8 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: 0xa3a3a3 } )
    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.position.z = -0.1;
    scene.add( plane );

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.autoClear = false;
    renderer.setSize( window.innerWidth, window.innerHeight);
    document.body.appendChild( renderer.domElement );

    // Stats & Resize Window & controller
    stats = new Stats();
    document.body.appendChild( stats.dom );

    window.addEventListener( 'resize', onWindowResize );

    const controls = new OrbitControls( camera, renderer.domElement ); 

    // Geometries & Material
    
    // Nodes

    entityGroup = makeNodes(entityGeometry, routerGeometry, pos, funcEdges, risk_mean, entityColors, extras, sizeMult, effectController.colorWithRisks); // Entity nodes and edges
    scene.add( entityGroup );
    
    
    // Edges
    // Connectivity

    edgeConnectivity = makeConnectivityEdges(edgeConnectivityMaterial, pos, funcEdges, risk_mean);
    
    scene.add( edgeConnectivity );
    edgeConnectivity.visible = effectController.showConnectivity;
    

      
    
}


function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    // Leave Legend Same Size
    orthoCamera.left = -2 * window.innerWidth / defWidth + 1;
    orthoCamera.top = 1 * window.innerHeight / defHeight;
    orthoCamera.bottom = -1 * window.innerHeight / defHeight;
    orthoCamera.updateProjectionMatrix();
    //console.log([window.innerWidth, window.innerHeight]);

    renderer.setSize( window.innerWidth, window.innerHeight );

}

function moveNodes(nodePos, stepSize=null, diamXY=1.3){
    
    let nodePosArray2d = tf.tidy( () =>  singleStepForceDirected(funcEdges, nodePos, stepSize, diamXY));
    const bounds = {upper:[2, 2], lower:[-2, -2]};
    nodePosArray2d = tf.tidy( () =>  scaleToBounds(nodePosArray2d, bounds) );
    // Cooling and convergence criteria right here

    edgePos = nodePos2edgePos(nodePosArray2d);
    edgeConnectivity.geometry.setAttribute( 'position', new THREE.BufferAttribute( edgePos, 3 ) );
    edgeConnectivity.geometry.attributes.position.needsUpdate = true;
    //Topology edges?

    setNodePos(entityGroup, nodePosArray2d);
    //console.log(tf.memory());

    return nodePosArray2d;
}

// Sets the node positions according to the new node position array
function setNodePos(nodeGroup, nodePos){
    const nNodes = nodeGroup.children.length;
    for ( let i = 0; i < nNodes; i ++ ) {

        const node = nodeGroup.children[i];
        node.position.set(nodePos[i][0], nodePos[i][1], 0);
    }
}

// Returns edge position buffer from node position array
function nodePos2edgePos(nodePosArr){
    const nNodes = nodePosArr.length;
    const edgePos = new Float32Array( 3 * 2 * nNodes * (nNodes - 1) );

    for (let i = 0; i < nNodes ; i++) {

        for (let j = 0; j < nNodes ; j++) {

            if (j == i){
                continue;
            }
            let k = i * nNodes + j;

            edgePos[ 6 * k ] = nodePosArr[i][0]; 
            edgePos[ 6 * k + 1] = nodePosArr[i][1];
            edgePos[ 6 * k + 2] = 0;

            edgePos[ 6 * k + 3] = nodePosArr[j][0]; 
            edgePos[ 6 * k + 4]  = nodePosArr[j][1];
            edgePos[ 6 * k + 5]  = 0;

        }
    }
    return edgePos;
}



function animate() {
    
    const time = Date.now() * 0.001; 
    
    if (effectController.activateForce){
        if ( counter % nFrame == 0) {
            nodePos = moveNodes(nodePos, stepSize, 2.3);
            //stepSize = (stepSize > dt) ? stepSize - dt : 0;
            
        }
        counter += 1;
    }

    renderer.clear();
	renderer.render( scene, camera );
    renderer.render( uiScene, orthoCamera );

    requestAnimationFrame( animate );

    stats.update();
}




