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
let uiScene, orthoCamera, sprite;

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
console.log(pos)

init();
animate();

function initGUI(){
    const options = {
        side: {
            FrontSide: THREE.FrontSide,
            BackSide: THREE.BackSide,
            DoubleSide: THREE.DoubleSide,
        },
        combine: {
            MultiplyOperation: THREE.MultiplyOperation,
            MixOperation: THREE.MixOperation,
            AddOperation: THREE.AddOperation,
        },
    }

    const gui = new GUI();

    gui.add( effectController, 'showConnectivity' ).onChange( function ( value ) {

        edgeConnectivity.visible = value

    } );

    const materialFolder = gui.addFolder('THREE.Material');
    materialFolder.add(nodeMaterial, 'transparent').onChange(() => nodeMaterial.needsUpdate = true);
    materialFolder.add(nodeMaterial, 'opacity', 0, 1, 0.01);
    materialFolder.add(nodeMaterial, 'depthTest');
    materialFolder.add(nodeMaterial, 'depthWrite');
    materialFolder.add(nodeMaterial, 'alphaTest', 0, 1, 0.01).onChange(() => updateMaterial());
    materialFolder.add(nodeMaterial, 'visible');
    materialFolder.add(nodeMaterial, 'side', options.side).onChange(() => updateMaterial());
    materialFolder.close();

    const materialData = {
        color: nodeMaterial.color.getHex(),
        emissive: nodeMaterial.emissive.getHex(),
        specular: nodeMaterial.specular.getHex()
    }

    const meshPhongMaterialFolder = gui.addFolder('THREE.MeshPhongMaterial');
    meshPhongMaterialFolder.addColor(materialData, 'color').onChange(() => {
        nodeMaterial.color.setHex(Number(materialData.color.toString().replace('#', '0x')))
    });
    meshPhongMaterialFolder.addColor(materialData, 'emissive').onChange(() => {
        nodeMaterial.emissive.setHex( Number(materialData.emissive.toString().replace('#', '0x')) )
    });
    meshPhongMaterialFolder.addColor(materialData, 'specular').onChange(() => {
        nodeMaterial.specular.setHex(Number(materialData.specular.toString().replace('#', '0x')))
    });
    meshPhongMaterialFolder.add(nodeMaterial, 'shininess', 0, 1024);
    meshPhongMaterialFolder.add(nodeMaterial, 'wireframe');
    meshPhongMaterialFolder.add(nodeMaterial, 'wireframeLinewidth', 0, 10);
    meshPhongMaterialFolder.add(nodeMaterial, 'flatShading').onChange(() => updateMaterial());
    meshPhongMaterialFolder.add(nodeMaterial, 'combine', options.combine).onChange(() => updateMaterial());
    meshPhongMaterialFolder.add(nodeMaterial, 'reflectivity', 0, 1);
    meshPhongMaterialFolder.add(nodeMaterial, 'refractionRatio', 0, 1);
    meshPhongMaterialFolder.open();

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
    //scene.background = new THREE.Color( 0xc4c4c4 );

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 8, 8 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: 0xa3a3a3 } )
    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.position.z = -0.1
    scene.add( plane );

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

function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

}

function updateMaterial() {
    //nodeMaterial.side = Number(material.side) as THREE.Side;
    //nodeMaterial.combine = Number(material.combine) as THREE.Combine;
    nodeMaterial.needsUpdate = true;
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

	renderer.render( scene, camera );

    requestAnimationFrame( animate );

    stats.update();
}




