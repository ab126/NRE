import * as THREE from 'three';

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import vertexShader from '../../shaders/vertex.glsl.js'
import fragmentShader from '../../shaders/fragment.glsl.js'

let camera, scene, renderer, stats;
let entityGroup;
let nodeColors, nodePosArray, nodeSizes;
let edgeConnectivity, edgeColors, edgePos;
let uiScene, orthoCamera, sprite, lut;

// Node parameters
const baseSize = 0.03; //0.05
const sizeMult = .07;
const nodeGeometry = new THREE.ConeGeometry( 0.1, 0.2, 3 );
const nodeMaterial = new THREE.MeshPhongMaterial();//new THREE.MeshStandardMaterial( {color: 0x0a0859} );

const effectController = {
    showConnectivity: true
};

// Read planar positions
//const {pos, topologyEdges, risk_mean, risk_cov, funcEdges, entityColors, extras} = data
let {pos, risk, edges, entityColors, extras} = generateSampleNet(0, 0 ,2);
const nNodes = Object.keys(pos).length;


init();
animate();

function initGUI(){

    const gui = new GUI();

    gui.add( effectController, 'showConnectivity' ).onChange( function ( value ) {

        edgeConnectivity.visible = value

    } );

}

function init(){ 
    
    initGUI();
    
    // Scene & Camera
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 4;
    scene.add(camera);

    // Legend
    generateLegend();

    // Lights
    scene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );
    scene.background = new THREE.Color( 0xc4c4c4 );

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 8, 8 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: 0xa3a3a3 } )
    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.position.z = -0.1
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
        degrees[i] = edges[i].reduce((acc, val) => acc + val );

        nodeColors[ i * 4 ] = effectController.colorWithRisks ? risk[name] / extras.diam_z : entityColors[name][0];
        nodeColors[ i * 4 + 1] = effectController.colorWithRisks ? 0 : entityColors[name][1];
        nodeColors[ i * 4 + 2] = effectController.colorWithRisks ? 0 : entityColors[name][2];
        nodeColors[ i * 4 + 3] = effectController.colorWithRisks ? 1 : entityColors[name][3];

    }
    const minDeg = Math.min(...degrees);
    const maxDeg = Math.max(...degrees);
    nodeSizes = Array(nNodes);
    
    
    for ( let i = 0; i < nNodes; i ++ ) {
        nodeSizes[i] = baseSize + sizeMult * (degrees[i] - minDeg) / (maxDeg - minDeg);        

        const node = new THREE.Mesh( nodeGeometry, nodeMaterial );
        node.rotateX(Math.PI /2);

        node.position.set(nodePosArray[i][0], nodePosArray[i][1], 0);
        //node.material.color.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2]);
        //node.material.color.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2]);
        entityGroup.add( node );
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

function generateLegend(){

    // Scene
    uiScene = new THREE.Scene();
    orthoCamera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, .1, 2 );
    orthoCamera.position.set( -0.8, 0, 1 );

    // Sprite
    sprite = new THREE.Sprite( new THREE.SpriteMaterial( { color:'#424242' } ) );
    sprite.scale.x = 0.4;
    uiScene.add( sprite );

    // Node
    const node = new THREE.Mesh( nodeGeometry, nodeMaterial );
    node.position.set(-0.1, 0.4, 0);
    node.scale.set(0.5, 0.5, 0.5);
    uiScene.add(node);

    // Edge 
    
}

function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

}

// Generate sample network at the given location
function generateSampleNet(xCenter, yCenter = 0, diam = 2){

    let pos = {a: [xCenter, yCenter], b: [xCenter, yCenter + diam/2], c: [xCenter - diam/2, yCenter - diam/2],
                d: [xCenter + diam/2, yCenter - diam/2]};
    let risk = {a: 0, b:1, c:2, d:3};
    let edges = [ [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]];
    let entityColors = {a:[0.8, 0.2, 1, 1], b:[0, 1, 1, 1], c:[0.2, 0.8, 1, 1], d:[0.6, 0.4, 1, 1]};
    let extras = {diam_z: 2, radius:1}

    return {pos, risk, edges, entityColors, extras}
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
            edgePositions[ 6 * k + 2] = elavate ? risk[src] : 0;

            edgeColors[ 8 * k ] = (edges[i][j])** (1/3) * 255 ; 
            edgeColors[ 8 * k + 1] = 0;
            edgeColors[ 8 * k + 2] = 0;
            edgeColors[ 8 * k + 3] = (edges[i][j]) ** (3) * 255;

            edgePositions[ 6 * k + 3] = pos[dst][0]; 
            edgePositions[ 6 * k + 4]  = pos[dst][1];
            edgePositions[ 6 * k + 5]  = elavate ? risk[dst] : 0;

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

    renderer.clear();
	renderer.render( scene, camera );
    renderer.render( uiScene, orthoCamera );

    requestAnimationFrame( animate );

    stats.update();
}




