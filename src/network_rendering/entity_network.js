import * as THREE from 'three';

import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import vertexShader from './shaders/vertex.glsl'
import fragmentShader from './shaders/fragment.glsl'

import * as data from './saves/net_data1.json' assert {type: 'json'}; 

let camera, scene, renderer, controls;
let entityGroup;
let nodeGeometry, nodePositions, nodePointCloud;
let edgeGeometry, edgePositions, edgeTopology;
let edgeConnectivity, edgeColors, allEdgePositions;
const radius = 0.05;
const dz = 0.01

const effectController = {
    solidEntities: true,
    showTopology: false,
    showRisks: false
};

// Read planar positions
const {pos, topologyEdges, risk_mean, risk_cov, funcEdges} = data
const nNodes = Object.keys(pos).length;
const nEdges = Object.keys(topologyEdges).length;

init();
animate();

function initGUI(){
    const gui = new GUI();

    gui.add( effectController, 'solidEntities' ).onChange( function ( value ) {

        entityGroup.visible = value;
        nodePointCloud.visible = !value

    } );

    gui.add( effectController, 'showTopology' ).onChange( function ( value ) {

        edgeTopology.visible = value;
        edgeConnectivity.visible = !value

    } );

    gui.add( effectController, 'showRisks' ).onChange( function ( elavate ) {
        
        for ( let i = 0; i < nNodes; i ++ ) {
            let name = Object.keys(pos)[i]
    
            nodePositions[ 3 * i + 2 ] = elavate ? risk_mean[name]: 0;

        }

        let src, dst;

        for (let i = 0; i < nEdges; i++) {

            [src, dst] = topologyEdges[i];

            edgePositions[ 6 * i + 2] = elavate ? risk_mean[src] : 0;

            edgePositions[ 6 * i + 5] = elavate ? risk_mean[dst] : 0;
        }

        for (let i = 0; i < nNodes ; i++) {

            for (let j = 0; j < nNodes ; j++) {
    
                if (j == i){
                    continue;
                }
                let k = i * nNodes + j;
                let src = Object.keys(pos)[i];
                let dst = Object.keys(pos)[j];
    
                allEdgePositions[ 6 * k + 2] = elavate ? risk_mean[src] : 0;
    
                allEdgePositions[ 6 * k + 5]  = elavate ? risk_mean[dst] : 0;

            }
        }
        // TODO: Need not update node positions additionally
        for ( let i = 0; i < nNodes; i ++ ) {

            let dodec = entityGroup.children[i]
            dodec.position.set(nodePositions[ 3 * i ], nodePositions[ 3 * i + 1 ], nodePositions[ 3 * i + 2])
            
        }

        nodePointCloud.geometry.attributes.position.needsUpdate = true;
        edgeTopology.geometry.attributes.position.needsUpdate = true;
        edgeConnectivity.geometry.attributes.position.needsUpdate = true;
        
        /*
        entityGroup.children.forEach(element => {
            element.geometry.attributes.position.needsUpdate = true;
        });*/

    } );
}

function init(){    

    // GUI
    initGUI();   
    
    // Scene & Camera
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 4;
    scene.add(camera)

    // Lights
    const light = new THREE.SpotLight( 0xffffff, 4.5 );
    light.position.set( 0, 0, 5 );
    //light.castShadow = true;
    scene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );
    scene.background = new THREE.Color( 0xc4c4c4 );
    //scene.add( light );

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 8, 8 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: 0xa3a3a3 } )
    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    //plane.position.z = -dz
    //plane.receiveShadow = true;
    scene.add( plane );
    console.log(plane)

    // Grid Lines
    const helper = new THREE.GridHelper( 5, 20 );
    helper.position.z = 0.01;
    helper.rotateX(Math.PI / 2)
    helper.material.opacity = 0.85;
    helper.material.transparent = false;
    scene.add( helper );

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize( window.innerWidth, window.innerHeight );
    //renderer.shadowMap.enabled = true;
    document.body.appendChild( renderer.domElement );

    // Geometries & Material
    
    // Nodes

    entityGroup = new THREE.Group(); // Entity nodes and edges
    scene.add( entityGroup );
    
    nodeGeometry = new THREE.BufferGeometry();
    nodePositions = new Float32Array( nNodes * 3 );

    const nodeMaterial = new THREE.MeshStandardMaterial( {color: 0x0a0859} );
    const nodePointMaterial = new THREE.PointsMaterial( {
        color: 0x242323,
        size: 4,
        //blending: THREE.AdditiveBlending,
        transparent: false,
        sizeAttenuation: false
    } );

    for ( let i = 0; i < nNodes; i ++ ) {
        let name = Object.keys(pos)[i]

        nodePositions[ i * 3 ] = pos[name][0];
        nodePositions[ i * 3 + 1 ] = pos[name][1];
        nodePositions[ i * 3 + 2 ] = 0;

    }
    
    nodeGeometry.setAttribute( 'position', new THREE.BufferAttribute( nodePositions, 3 ).setUsage( THREE.DynamicDrawUsage ) );
        
    nodePointCloud = new THREE.Points( nodeGeometry, nodePointMaterial );
    nodePointCloud.visible = false
    scene.add(nodePointCloud)
    
    for ( let i = 0; i < nNodes; i ++ ) {
        const geometryDodec = new THREE.DodecahedronGeometry( radius );
        const dodec = new THREE.Mesh( geometryDodec, nodeMaterial );

        /*dodec.position.x = nodePositions[ 3 * i ]
        dodec.position.y = nodePositions[ 3 * i  + 1]
        dodec.position.z = nodePositions[ 3 * i + 2]*/
        dodec.position.set(nodePositions[ 3 * i ], nodePositions[ 3 * i + 1 ], nodePositions[ 3 * i + 2])
        
        entityGroup.add( dodec );
    }

    // Edges

    makeEdgePositions(topologyEdges, false);
    
    // Network Topology

    const edgeTopologyGeometry = new THREE.BufferGeometry()
    edgeTopologyGeometry.setAttribute( 'position', new THREE.BufferAttribute( edgePositions, 3 ) );
    
    const edgeTopologyMaterial = new THREE.LineBasicMaterial({
        color: 0xb08102
    });
    
    edgeTopology = new THREE.LineSegments( edgeTopologyGeometry, edgeTopologyMaterial );
    edgeTopology.visible = false
    scene.add( edgeTopology );

    // Connectivity
    makeAllEdges(false);

    const edgeConnectivityGeometry = new THREE.BufferGeometry()
    edgeConnectivityGeometry.setAttribute( 'position', new THREE.BufferAttribute( allEdgePositions, 3 ) );
    edgeConnectivityGeometry.setAttribute( 'color', new THREE.Uint8BufferAttribute( edgeColors, 4, true ) );

    /*
    const edgeConnectivityMaterial = new THREE.LineBasicMaterial({
        //color: 0xbf0000,
        vertexColors: true,
        //transparent: true
    });*/
    
    const edgeConnectivityMaterial = new THREE.ShaderMaterial( {
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        transparent: true,
    } );

    edgeConnectivity = new THREE.LineSegments( edgeConnectivityGeometry, edgeConnectivityMaterial );
    scene.add( edgeConnectivity );

    const controls = new OrbitControls( camera, renderer.domElement );   
    
}

/**
 * Makes edge defining points
 * @param {*} src Source entity
 * @param {*} dst Destination entity
 */
function makeEdgePositions(edges, elavate){
    edgePositions = new Float32Array( nEdges * 2 * 3 );
    edgeColors = new Float32Array( nEdges * 2 * 3 );
    let src, dst;

    for (let i = 0; i < edges.length; i++) {

        [src, dst] = edges[i];

        edgePositions[ 6 * i ] = pos[src][0]; 
        edgePositions[ 6 * i + 1] = pos[src][1];
        edgePositions[ 6 * i + 2] = elavate ? risk_mean[src] : 0;

        edgeColors[ 6 * i ] = Math.random() * 255; 
        edgeColors[ 6 * i + 1] = 0;
        edgeColors[ 6 * i + 2] = 0;

        edgePositions[ 6 * i + 3] = pos[dst][0]; 
        edgePositions[ 6 * i + 4] = pos[dst][1];
        edgePositions[ 6 * i + 5] = elavate ? risk_mean[dst] : 0;

        edgeColors[ 6 * i + 3] = edgeColors[ 6 * i ]; 
        edgeColors[ 6 * i + 4] = 0;
        edgeColors[ 6 * i + 5] = 0;

    }
}

function makeAllEdges(elavate) {
    allEdgePositions = new Float32Array( 3 * 2 * nNodes * (nNodes - 1) );
    edgeColors = new Float32Array( 4 * 2 * nNodes * (nNodes - 1) );

    for (let i = 0; i < nNodes ; i++) {

        for (let j = 0; j < nNodes ; j++) {

            if (j == i){
                continue;
            }
            let k = i * nNodes + j;
            let src = Object.keys(pos)[i];
            let dst = Object.keys(pos)[j];

            allEdgePositions[ 6 * k ] = pos[src][0]; 
            allEdgePositions[ 6 * k + 1] = pos[src][1];
            allEdgePositions[ 6 * k + 2] = elavate ? risk_mean[src] : 0;

            edgeColors[ 8 * k ] = (funcEdges[i][j])** (1/3) * 255 ; 
            edgeColors[ 8 * k + 1] = 0;
            edgeColors[ 8 * k + 2] = 0;
            edgeColors[ 8 * k + 3] = (funcEdges[i][j]) ** (3) * 255;

            allEdgePositions[ 6 * k + 3] = pos[dst][0]; 
            allEdgePositions[ 6 * k + 4]  = pos[dst][1];
            allEdgePositions[ 6 * k + 5]  = elavate ? risk_mean[dst] : 0;

            edgeColors[ 8 * k + 4] = edgeColors[ 8 * k ]; 
            edgeColors[ 8 * k + 5] = edgeColors[ 8 * k + 1];
            edgeColors[ 8 * k + 6] = edgeColors[ 8 * k + 2];
            edgeColors[ 8 * k + 7] = edgeColors[ 8 * k + 3];
        }
    }
}

function animate() {
    
    const time = Date.now() * 0.001;

	renderer.render( scene, camera );

    requestAnimationFrame( animate );
}




