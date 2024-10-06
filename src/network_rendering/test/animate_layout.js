import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';

import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import vertexShader from '../shaders/vertex.glsl.js'
import fragmentShader from '../shaders/fragment.glsl.js'

import * as data from '../saves/net_data_small.json' assert {type: 'json'}; // 63
import {singleStepForceDirected} from './force-directed.js'

console.log(data)

let camera, scene, renderer;
let entityGroup;
let nodeColors, nodePosArray;
let edgeConnectivity, edgeColors, edgePos;
const radius = 0.05;

const effectController = {
    showConnectivity: true,
    elevateWithRisks: false,
    colorWithRisks: false,
    maxIter: 5000,
    activateForce: false
};

// Read planar positions
const {pos, topologyEdges, risk_mean, risk_cov, funcEdges, entityColors, extras} = data
const nNodes = Object.keys(pos).length;

let nodePos = Array.from(Object.keys(pos), (key) => pos[key]);
let stepSize = .01;
let dt = stepSize / (effectController.maxIter + 1);

const nFrame = 10;
let counter = 0;

init();
animate();

function initGUI(){
    const gui = new GUI();

    gui.add( effectController, 'showConnectivity' ).onChange( function ( value ) {

        edgeConnectivity.visible = value

    } );

    //Maybe better to update whats changed?
    gui.add( effectController, 'elevateWithRisks' ).onChange( function ( elavate ) {
        

        //Update edgePositions
        for (let i = 0; i < nNodes ; i++) {

            for (let j = 0; j < nNodes ; j++) {
    
                if (j == i){
                    continue;
                }
                let k = i * nNodes + j;
                let src = Object.keys(pos)[i];
                let dst = Object.keys(pos)[j];
    
                edgePos[ 6 * k + 2] = elavate ? risk_mean[src] : 0;
    
                edgePos[ 6 * k + 5]  = elavate ? risk_mean[dst] : 0;

            }
        }

        // TODO: Need not update node positions additionally
        for ( let i = 0; i < nNodes; i ++ ) {
            let name = Object.keys(pos)[i];
            let dodec = entityGroup.children[i];

            dodec.position.set(nodePosArray[i][0], nodePosArray[i][1], elavate ? risk_mean[name]: 0)
            
        }

        edgeConnectivity.geometry.attributes.position.needsUpdate = true;
        
        /*
        entityGroup.children.forEach(element => {
            element.geometry.attributes.position.needsUpdate = true;
        });*/
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

                let dodec = entityGroup.children[i];
                dodec.material.color.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2]);
        
            }
        }
    } );

    gui.add( effectController, 'maxIter', 10, 100, 10).onChange( function ( value ){
        dt = stepSize / (value + 1);
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

    // Geometries & Material
    
    // Nodes

    entityGroup = new THREE.Group(); // Entity nodes and edges
    scene.add( entityGroup );
    
    nodeColors = new Float32Array( nNodes * 4 );
    nodePosArray = Array(nNodes);
    

    for ( let i = 0; i < nNodes; i ++ ) {
        let name = Object.keys(pos)[i]

        nodePosArray[i] = pos[name];

        nodeColors[ i * 4 ] = entityColors[name][0];
        nodeColors[ i * 4 + 1] = entityColors[name][1];
        nodeColors[ i * 4 + 2] = entityColors[name][2];
        nodeColors[ i * 4 + 3] = entityColors[name][3];

    }
    
    
    for ( let i = 0; i < nNodes; i ++ ) {
        const geometryDodec = new THREE.DodecahedronGeometry( radius );
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

function moveNodes(nodePos, stepSize=null){
    
    const nodePosArray2d = tf.tidy( () =>  singleStepForceDirected(funcEdges, nodePos, stepSize));
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
 * Makes edge defining points from edge topology object
 * @param {*} edges Array whose entries are edges with ['source', 'destination']
 * @param {*} pos Object whose fields are entity ids with [pos_x, pos_y] array describing 2D position
 * @param {*} elavate If true, elavates the entities according the mean value of their risks in z direction
 * @returns [edgeTopologyPositions]
 *  edgeTopologyPositions: Float32Array of source position coordinates and destination  position coordinates
 */
function makeEdgePositions(edges, pos, elavate){
    const nEdges = Object.keys(edges).length;
    const edgeTopologyPositions = new Float32Array( nEdges * 2 * 3 );
    let src, dst;

    for (let i = 0; i < edges.length; i++) {

        [src, dst] = edges[i];

        edgeTopologyPositions[ 6 * i ] = pos[src][0]; 
        edgeTopologyPositions[ 6 * i + 1] = pos[src][1];
        edgeTopologyPositions[ 6 * i + 2] = elavate ? risk_mean[src] : 0;
        edgeTopologyPositions[ 6 * i + 3] = pos[dst][0]; 
        edgeTopologyPositions[ 6 * i + 4] = pos[dst][1];
        edgeTopologyPositions[ 6 * i + 5] = elavate ? risk_mean[dst] : 0;

        return [edgeTopologyPositions];
    }
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


function makeOutline( scene ) {
    const pieOutline = new THREE.Group();
    scene.add(pieOutline);

    const outlineMaterial = new THREE.LineBasicMaterial( { color: 0xffffff } );

    const outerCircle = new THREE.EllipseCurve(
        0,  0,            // ax, aY
        extras.radius + extras.diam_xy/2, extras.radius + extras.diam_xy/2,  // xRadius, yRadius
        0,  2 * Math.PI,  // aStartAngle, aEndAngle
        false,            // aClockwise
        0                 // aRotation
    );
    let points = outerCircle.getPoints( 1000 );
    let pieGeometry = new THREE.BufferGeometry().setFromPoints( points );
    const outerLine = new THREE.Line( pieGeometry, outlineMaterial );
    pieOutline.add(outerLine);

    const innerCircle = new THREE.EllipseCurve(
        0,  0,            // ax, aY
        extras.radius - extras.diam_xy/2, extras.radius - extras.diam_xy/2,  // xRadius, yRadius
        0,  2 * Math.PI,  // aStartAngle, aEndAngle
        false,            // aClockwise
        0                 // aRotation
    );
    points = innerCircle.getPoints( 1000 );
    pieGeometry = new THREE.BufferGeometry().setFromPoints( points );
    const innerLine = new THREE.Line( pieGeometry, outlineMaterial );
    pieOutline.add(innerLine);

    // The Line Segments

    const n_segments = extras.n_cluster - 1;

    for (let i = 0; i < n_segments ; i++) {
        
        const phi = 2 * Math.PI / n_segments;
        let nu_i = phi * i + phi / 2;

        let points = [];
        points.push( new THREE.Vector3( (extras.radius - extras.diam_xy/2) * Math.sin( nu_i),
                                        (extras.radius - extras.diam_xy/2) * Math.cos( nu_i), 0 ) );
        points.push( new THREE.Vector3( (extras.radius + extras.diam_xy/2) * Math.sin( nu_i),
                                        (extras.radius + extras.diam_xy/2) * Math.cos( nu_i), 0 ) );

        const geometry = new THREE.BufferGeometry().setFromPoints( points );
        const line = new THREE.Line( geometry, outlineMaterial );
        pieOutline.add(line);
    }
    return pieOutline
}

function animate() {
    
    const time = Date.now() * 0.001;

    if (effectController.activateForce){
        if ( counter % nFrame == 0) {
            nodePos = moveNodes(nodePos, stepSize);
            stepSize = (stepSize > dt) ? stepSize - dt : 0;
            
        }
        counter += 1;
    }
    

	renderer.render( scene, camera );

    requestAnimationFrame( animate );
}




