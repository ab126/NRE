import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import vertexShader from '../../shaders/vertex.glsl.js'
import fragmentShader from '../../shaders/fragment.glsl.js'

import * as data from '../../saves/net_data_medium2.json' assert {type: 'json'}; // 63
import {singleStepForceDirected, scaleToBounds} from './force-directed.js'

console.log(data);

let camera, scene, renderer, stats;
let entityGroup;
let nodeColors, nodePosArray, nodeSizes;
let edgeConnectivity, edgeColors, edgePos;

// Node parameters
const baseSize = 0.03; //0.05
const sizeMult = .07;

const effectController = {
    showConnectivity: true,
    colorWithRisks: true,
    maxIter: 50,
    stepSize: .1,
    activateForce: false
};

// Read planar positions
const {pos, topologyEdges, risk_mean, risk_cov, funcEdges, entityColors, extras} = data
const nNodes = Object.keys(pos).length;

let nodePos = Array.from(Object.keys(pos), (key) => pos[key]);
let stepSize = .01;
let dt = stepSize / (effectController.maxIter + 1);

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

        if (value == true) {
            for ( let i = 0; i < nNodes; i ++ ) {

                let name = Object.keys(pos)[i];
                let dodec = entityGroup.children[i];

                dodec.material.color.setRGB( risk_mean[name] / extras.diam_z , 0, 0);
            
            }
        } else {
            for ( let i = 0; i < nNodes; i ++ ) {
                let name = Object.keys(pos)[i];
                let dodec = entityGroup.children[i];
                dodec.material.color.setRGB(entityColors[name][0], entityColors[name][1], entityColors[name][2]);
        
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
    scene.add(camera)

    // Lights
    scene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );
    scene.background = new THREE.Color( 0xc4c4c4 );

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 8, 8 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: 0xa3a3a3 } )
    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    scene.add( plane );

    // Pie Outline
    //const pieOutline = makeOutline(scene);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    // Stats & Resize Window
    stats = new Stats();
    document.body.appendChild( stats.dom );

    window.addEventListener( 'resize', onWindowResize );

    // Geometries & Material
    
    // Nodes

    entityGroup = new THREE.Group(); // Entity nodes and edges
    scene.add( entityGroup );
    
    nodeColors = new Float32Array( nNodes * 4 );
    nodePosArray = Array(nNodes);
    const degrees = Array(nNodes);
    

    for ( let i = 0; i < nNodes; i ++ ) {
        let name = Object.keys(pos)[i];

        nodePosArray[i] = pos[name];
        degrees[i] = funcEdges[i].reduce((acc, val) => acc + val );

        nodeColors[ i * 4 ] = effectController.colorWithRisks ? risk_mean[name] / extras.diam_z : entityColors[name][0];
        nodeColors[ i * 4 + 1] = effectController.colorWithRisks ? 0 : entityColors[name][1];
        nodeColors[ i * 4 + 2] = effectController.colorWithRisks ? 0 : entityColors[name][2];
        nodeColors[ i * 4 + 3] = effectController.colorWithRisks ? 1 : entityColors[name][3];

    }
    const minDeg = Math.min(...degrees);
    const maxDeg = Math.max(...degrees);
    nodeSizes = Array(nNodes);
    
    
    for ( let i = 0; i < nNodes; i ++ ) {
        nodeSizes[i] = baseSize + sizeMult * (degrees[i] - minDeg) / (maxDeg - minDeg);
        const geometryDodec = new THREE.DodecahedronGeometry( nodeSizes[i] );
        const nodeMaterial = new THREE.MeshStandardMaterial( {color: 0x0a0859} );

        const dodec = new THREE.Mesh( geometryDodec, nodeMaterial );

        dodec.position.set(nodePosArray[i][0], nodePosArray[i][1], 0);
        dodec.material.color.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2]);
        entityGroup.add( dodec );
    }
    

    // Edges

    // Connectivity
    [edgePos, edgeColors] = makeAllEdges(pos, false);

    const edgeConnectivityGeometry = new THREE.BufferGeometry();
    edgeConnectivityGeometry.setAttribute( 'position', new THREE.BufferAttribute( edgePos, 3 ) );
    edgeConnectivityGeometry.setAttribute( 'color', new THREE.Uint8BufferAttribute( edgeColors, 4, true ) );
    
    const edgeConnectivityMaterial = new THREE.ShaderMaterial( {
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        transparent: true,
    } );

    edgeConnectivity = new THREE.LineSegments( edgeConnectivityGeometry, edgeConnectivityMaterial );
    scene.add( edgeConnectivity );
    edgeConnectivity.visible = effectController.showConnectivity;
    

    const controls = new OrbitControls( camera, renderer.domElement );   
    
}

function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

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

        const dodec = nodeGroup.children[i];
        dodec.position.set(nodePos[i][0], nodePos[i][1], 0);
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


/**
 * Makes edge defining points from functional connectivity edge array
 * @param {*} pos Object whose fields are entity ids with [pos_x, pos_y] array describing 2D position
 * @param {*} elavate If true, elavates the entities according the mean value of their risks in z direction
 * @returns [edgeConnectivityPositions, edgeColors]
 *  edgeConnectivityPositions: Float32Array of source position coordinates and destination  position coordinates
 *  edgeColors: Float32Array of RGBA values of edges
 */
function makeAllEdges(pos, elavate) {
    const nNodes = Object.keys(pos).length;
    const edgePositions = new Float32Array( 3 * 2 * nNodes * (nNodes - 1) );
    const edgeColors = new Float32Array( 4 * 2 * nNodes * (nNodes - 1) );

    for (let i = 0; i < nNodes ; i++) {

        for (let j = 0; j < nNodes ; j++) {

            if (j == i){
                continue;
            }
            let k = i * nNodes + j;
            let src = Object.keys(pos)[i];
            let dst = Object.keys(pos)[j];

            edgePositions[ 6 * k ] = pos[src][0]; 
            edgePositions[ 6 * k + 1] = pos[src][1];
            edgePositions[ 6 * k + 2] = elavate ? risk_mean[src] : 0;

            edgeColors[ 8 * k ] = (funcEdges[i][j])** (1/3) * 255 ; 
            edgeColors[ 8 * k + 1] = 0;
            edgeColors[ 8 * k + 2] = 0;
            edgeColors[ 8 * k + 3] = (funcEdges[i][j]) ** (3) * 255;

            edgePositions[ 6 * k + 3] = pos[dst][0]; 
            edgePositions[ 6 * k + 4]  = pos[dst][1];
            edgePositions[ 6 * k + 5]  = elavate ? risk_mean[dst] : 0;

            edgeColors[ 8 * k + 4] = edgeColors[ 8 * k ]; 
            edgeColors[ 8 * k + 5] = edgeColors[ 8 * k + 1];
            edgeColors[ 8 * k + 6] = edgeColors[ 8 * k + 2];
            edgeColors[ 8 * k + 7] = edgeColors[ 8 * k + 3];
        }
    }
    return [edgePositions, edgeColors];
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

    requestAnimationFrame( animate );

    stats.update();
}




