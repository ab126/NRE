import * as THREE from 'three';

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';

import vertexShader from '../../shaders/vertex.glsl.js'
import fragmentShader from '../../shaders/fragment.glsl.js'

let camera, scene, renderer, stats;
let entityGroup;
let nodeColors, nodePosArray, nodeSizes;
let edgeConnectivity, edgeColors, edgePos;
let uiScene, orthoCamera, font;

// Legend Parameters
const defWidth = 900; 
const defHeight = 500; 

// Node & Edge Parameters
const baseSize = 0.03; //0.05
const sizeMult = .07;
const nodeGeometry = new THREE.ConeGeometry( 0.1, 0.2, 3 );
const nodeMaterial = new THREE.MeshPhongMaterial();//new THREE.MeshStandardMaterial( {color: 0x0a0859} );

const edgeConnectivityMaterial = new THREE.ShaderMaterial( {
    vertexShader: vertexShader,
    fragmentShader: fragmentShader,
    transparent: true,
} );

const edgeMaterial = new THREE.LineBasicMaterial({
    color: '#ff2929'
});

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
    

    edgeConnectivity = new THREE.LineSegments( edgeConnectivityGeometry, edgeConnectivityMaterial );
    scene.add( edgeConnectivity );
    edgeConnectivity.visible = effectController.showConnectivity;
    

    const controls = new OrbitControls( camera, renderer.domElement );   
    
}



function generateLegend(){

    // Scene
    uiScene = new THREE.Scene();
    orthoCamera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, .1, 2 );
    //orthoCamera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    orthoCamera.position.set( -0.8, 0, 1 );
    orthoCamera.left = -2 * window.innerWidth / defWidth + 1;
    orthoCamera.top = 1 * window.innerHeight / defHeight;
    orthoCamera.bottom = -1 * window.innerHeight / defHeight;
    orthoCamera.updateProjectionMatrix();

    // Organization
    const legend = new THREE.Group();
    const legendHeader = new THREE.Group();
    const nodeSegment = new THREE.Group();
    const connectivitySegment = new THREE.Group();
    const riskSegment = new THREE.Group();

    legend.add(legendHeader);
    legend.add(nodeSegment);
    legend.add(connectivitySegment);
    legend.add(riskSegment);
    uiScene.add(legend);

    // Sprite
    // 0,0 is the center
    const sprite = new THREE.Sprite( new THREE.SpriteMaterial( { color:'#424242' } ) );
    sprite.scale.x = 0.4;
    sprite.position.set(0, 0, 0);
    legend.add( sprite );

    // Text Parameters

    const textYShift = -0.02;

    const headerMat = new THREE.MeshBasicMaterial( {
        color: 0xffffff,
        transparent: true,
        opacity: 1,
        side: THREE.DoubleSide
    } );

    const liteMat = new THREE.MeshBasicMaterial( {
        color: 0xffffff,
        transparent: true,
        opacity: .8,
        side: THREE.DoubleSide
    } );

    // Header
    addFont("Legend", [ -.09, .42, 0], headerMat, legendHeader);
    
    const pts = new Float32Array([-.19, .39, 0,  .17, .39, 0 ]);
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute( 'position', new THREE.BufferAttribute( pts, 3 ).setUsage( THREE.DynamicDrawUsage ) );
    const line = new THREE.Line(lineGeometry, headerMat);
    legendHeader.add(line);

    // Node Segment
    const [xPos1, yPos1] = legendSegmentLocations(2, 0.3, 0.2);

    const entitySample = new THREE.Mesh( nodeGeometry, nodeMaterial );
    entitySample.position.set(xPos1[1], yPos1[1], 0); // -0.13, 0.25
    entitySample.scale.set(0.5, 0.5, 0.5);
    nodeSegment.add( entitySample );

    addFont(": Entities", [ xPos1[2], yPos1[1] + textYShift, 0], liteMat, nodeSegment); // -.09, .22

    // Connectivity Segment
    const [ptsShiftx, ptsShifty] = [0.025, 0.05];
    const [xPos2, yPos2] = legendSegmentLocations(2, 0, 0.2);
    console.log(xPos2, yPos2);
    
    const pointMaterial = new THREE.PointsMaterial( {
        color: 0x242323,
        size: 4,//4
        //blending: THREE.AdditiveBlending,
        transparent: false,
        sizeAttenuation: false
    } );

    const edgeSample = new THREE.Group();
    connectivitySegment.add(edgeSample);

    const pointGeometry = new THREE.BufferGeometry();
    const edgePointPositions = new Float32Array([xPos2[1] - ptsShiftx, yPos2[1] - ptsShifty, .01, xPos2[1] + ptsShiftx, yPos2[1] + ptsShifty, .01 ]); //[-.16, .0, .1, -.11, .10, .1 ]
    pointGeometry.setAttribute( 'position', new THREE.BufferAttribute( edgePointPositions, 3 ).setUsage( THREE.DynamicDrawUsage ) );
    const edgeSamplePoints = new THREE.Points( pointGeometry, pointMaterial );
    edgeSample.add(edgeSamplePoints);

    const edgeGeometry = new THREE.BufferGeometry()
    edgeGeometry.setAttribute( 'position', new THREE.BufferAttribute( edgePointPositions, 3 ) );    
    const edge = new THREE.LineSegments( edgeGeometry, edgeMaterial );
    edgeSample.add(edge);

    addFont(": Functional\n Connectivity", [xPos2[2], yPos2[1] + textYShift, 0], liteMat, connectivitySegment); //  -.09, .04, 0

    // Risk Segment
    const [xPos3, yPos3] = legendSegmentLocations(2, -0.2, 0.2);

    const colorNode = new THREE.Mesh( nodeGeometry, nodeMaterial );
    colorNode.position.set(-0.13, -.14, 0);
    colorNode.scale.set(0.5, 0.5, 0.5);
    //riskSegment.add(colorNode);
    addFont(": Risk\n Mean", [ xPos3[2], yPos3[1] + textYShift, 0], liteMat, riskSegment); // -.09, -.14, 0

}

/*
*/
/** Return the layout location of each row in the segment and preset horizontal coords.
 *      Segment is centered
 *      
 * @param {*} nRows Number of rows in a segment
 * @param {*} ySegment y coordinate of the segment center
 * @param {*} widthPerc Width of the segment divided by the legend width
 * @param {*} heightPerc Height of the segment divided by the legend height
 * @param {*} legendWidth Width of the legend box
 * @param {*} legendHeight Height of the legend box
 */
function legendSegmentLocations(nRows, ySegment, heightPerc, widthPerc = 0.9, legendWidth=0.4, legendHeight=1){
    const lineHeight = legendHeight * heightPerc / nRows;
    const startYPos = ySegment + legendHeight * heightPerc / 2 - lineHeight / 2;
    const yPos = Array.from({length: nRows}, (_, i) => startYPos - lineHeight * i);
    const xPos = Array( -widthPerc * legendWidth * 0.5, -widthPerc * legendWidth * 0.3, -widthPerc * legendWidth * 0.2);

    return [xPos, yPos]
}

function addFont(message, textPos, textMaterial, group2Add){

    const loader = new FontLoader();

    loader.load( 'fonts/helvetiker_regular.typeface.json', function ( font ) {

        const shapes = font.generateShapes(message, .04);
        const geometry = new THREE.ShapeGeometry( shapes );
        const text = new THREE.Mesh( geometry, textMaterial );
        text.scale.set(0.6, 1, 1);
        text.position.set(...textPos); 
        group2Add.add( text );

        animate();

    } );
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




