import * as THREE from 'three';

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { fontManager} from '../legend/legendMaker.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';
import TWEEN from '@tweenjs/tween.js'


let persCamera, camera, cameraHelper, scene, renderer, stats;
let entityGroup;
let nodeColors, nodePosArray, nodeSizes;
let edgeConnectivity, edgeColors, edgePos;
let uiScene, orthoCamera, sprite;
let plane, greeter;

const fontPath = 'fonts/helvetiker_regular.typeface.json';

// Node parameters
const baseSize = 0.03; //0.05
const sizeMult = .07;
const nodeGeometry = new THREE.OctahedronGeometry( 0.07, 4 );
const nodeMaterial = new THREE.MeshPhongMaterial({
    color:'#2CF604',
    emissive:'#000000',
    emissiveIntensity:1,
    specular:'#ffffff',
    shininess:30
});

const uniforms = {
    time : { value: 0.0 },
};

const edgeConnectivityMaterial = new THREE.ShaderMaterial( {
    uniforms: uniforms,
    vertexShader: document.getElementById( 'vertexShader' ).textContent,
    fragmentShader: document.getElementById( 'fragmentShader' ).textContent,
    blending: THREE.AdditiveBlending,
    transparent: true,
} );



/*
const edgeConnectivityMaterial = new THREE.LineBasicMaterial( {
    color: '#830101',
    linewidth: 1,
    linecap: 'round', //ignored by WebGLRenderer
    linejoin:  'round' //ignored by WebGLRenderer
} );
*/


const effectController = {
    showConnectivity: true,
    animateCamera: false
};

// Read planar positions
//const {pos, topologyEdges, risk_mean, risk_cov, funcEdges, entityColors, extras} = data
let {pos, risk, edges, entityColors, extras} = generateSampleNet(0, 0 ,2);
const nNodes = Object.keys(pos).length;

initGUI();
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

        edgeConnectivity.visible = value;

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
    
    // Scene 
    scene = new THREE.Scene();

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 8, 8 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: '#4a4a4a'} )
    plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.position.z = -0.1
    scene.add( plane );

    // Cameras    
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.set(0, -15, 2);
    camera.lookAt(plane.position);
    camera.translateY(5);
    
    scene.add(camera);

    persCamera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.1, 1000 );
    persCamera.position.set(0, 0, 4);
    console.log(persCamera.rotation);
    scene.add(persCamera);

    cameraHelper = new THREE.CameraHelper( persCamera );
    scene.add( cameraHelper );

    // Greeter
    const loader = new FontLoader();

    const liteMat = new THREE.MeshBasicMaterial( {
        color: 0xffffff,
        transparent: true,
        opacity: .8,
        side: THREE.DoubleSide
    } );
    const fontSize = 5;

    loader.load( fontPath, function ( font ) {
    
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

        const geometryDesc = new TextGeometry( 'Click Anywhere'.italics(), { 
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

        persCamera.position.set(0, -15, 20);
        persCamera.lookAt(greeter.position);
        persCamera.position.set(0, -15, 15)
        console.log(greeter.position);
    } );

    document.addEventListener('click', panToPlane);

    // Lights
    scene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );
    //scene.background = new THREE.Color( 0xc4c4c4 );
    const light = new THREE.DirectionalLight( 0xffffff, 0.5 );
    light.position.set(1, 1, 1);
    scene.add( light );

    

    // Renderer
    const container = document.getElementById('container')
    
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    container.appendChild( renderer.domElement );
    renderer.setScissorTest( true );

    // Stats & Resize Window
    stats = new Stats();
    container.appendChild( stats.dom );

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

    //const controls = new OrbitControls( persCamera, renderer.domElement );   
    //controls.target.set(0, 0, 0);
}

// Pans the camera to the initial plane
function panToPlane() {
    // Create a tweens
    const myEasing = TWEEN.Easing.Sinusoidal.In;

    new TWEEN.Tween(persCamera.position).to({
        x: 0,
        y: 0,
        z: 4
    }, 1000).easing(myEasing).start();

    new TWEEN.Tween(persCamera.rotation).to({
        x: 0,
    }, 1000).easing(myEasing).start();        
    persCamera.lookAt(0, 0, 0);

    document.removeEventListener('click', panToPlane);
}

function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    persCamera.aspect = window.innerWidth / window.innerHeight;
    persCamera.updateProjectionMatrix();

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
      
    requestAnimationFrame( animate );

    TWEEN.update();

	render();    

    stats.update();
}

function render() {

    uniforms.time.value += 0.01;
    

    cameraHelper.visible = true;
    renderer.setClearColor( 0x111111, 1 );
    renderer.setScissor( window.innerWidth /4, window.innerHeight /2, window.innerWidth /2, window.innerHeight /2);
    renderer.setViewport( window.innerWidth /4, window.innerHeight /2, window.innerWidth /2, window.innerHeight /2);	
    renderer.render( scene, camera );

    cameraHelper.visible = false;
    renderer.setClearColor( 0x000000, 1 );
    renderer.setScissor( window.innerWidth /4, 0, window.innerWidth /2, window.innerHeight /2);
    renderer.setViewport( window.innerWidth /4, 0, window.innerWidth /2, window.innerHeight /2);
				
    renderer.render( scene, persCamera );
    
}




