import * as THREE from 'three';

import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import * as data from './saves/net_data2.json' assert {type: 'json'}; 

let camera, scene, renderer, controls;
let entityGroup;
let nodeGeometry, nodePositions, nodePointCloud;
let edgeGeometry, edgePositions, edgeCloud;
const radius = 0.05;
const dz = 0.01

const effectController = {
    solidEntities: true,
    showLines: false,
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

    gui.add( effectController, 'showLines' ).onChange( function ( value ) {

        edgeCloud.visible = value;

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

        // TODO: Need not update node positions additionally
        for ( let i = 0; i < nNodes; i ++ ) {

            let dodec = entityGroup.children[i]
            dodec.position.set(nodePositions[ 3 * i ], nodePositions[ 3 * i + 1 ], nodePositions[ 3 * i + 2])
            
        }

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
    light.castShadow = true;
    scene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );
    scene.add( light );

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 20, 20 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: 0xd4d4d4 } )
    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.position.z = -dz
    plane.receiveShadow = true;
    scene.add( plane );

    // Grid Lines
    const helper = new THREE.GridHelper( 5, 10 );
    helper.position.z = 0.01;
    helper.rotateX(Math.PI / 2)
    helper.material.opacity = 0.25;
    helper.material.transparent = true;
    scene.add( helper );

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize( window.innerWidth, window.innerHeight );
    renderer.shadowMap.enabled = true;
    document.body.appendChild( renderer.domElement );

    // Geometries & Material
    
    // Nodes

    entityGroup = new THREE.Group(); // Entity nodes and edges
    scene.add( entityGroup );
    
    nodeGeometry = new THREE.BufferGeometry();
    nodePositions = new Float32Array( nNodes * 3 );

    const nodeMaterial = new THREE.MeshStandardMaterial( {color: 0x242323} );
    const nodePointMaterial = new THREE.PointsMaterial( {
        color: 0xFFFFFF,
        size: 2,
        blending: THREE.AdditiveBlending,
        transparent: true,
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

    console.log(entityGroup)
    console.log(nodePointCloud)
    
    /*
    Object.keys(pos).forEach(element => {
        entityGroup.add(makeNode(element, nodeMaterial, radius))
    });*/

    // Edges

    const edgeMaterial = new THREE.LineBasicMaterial({
        color: 0xffffff
    });

    makeEdgePositions(topologyEdges, false)
    //console.log(edgePositions)
    
    const edgeGeometry = new THREE.BufferGeometry()
    edgeGeometry.setAttribute( 'position', new THREE.BufferAttribute( edgePositions, 3 ) );
    
    edgeCloud = new THREE.LineSegments( edgeGeometry, edgeMaterial );
    edgeCloud.visible = false
    scene.add( edgeCloud );

    const controls = new OrbitControls( camera, renderer.domElement );   
    
}

/**
 * Makes edge defining points
 * @param {*} src Source entity
 * @param {*} dst Destination entity
 */
function makeEdgePositions(edges, elavate){
    edgePositions = new Float32Array( nEdges * 2 * 3 );
    let src, dst;

    for (let i = 0; i < edges.length; i++) {

        [src, dst] = edges[i];

        edgePositions[ 6 * i ] = pos[src][0]; 
        edgePositions[ 6 * i + 1] = pos[src][1];
        edgePositions[ 6 * i + 2] = elavate ? risk_mean[src] : 0;

        edgePositions[ 6 * i + 3] = pos[dst][0]; 
        edgePositions[ 6 * i + 4] = pos[dst][1];
        edgePositions[ 6 * i + 5] = elavate ? risk_mean[dst] : 0;
    }
}

function animate() {
    
    nodePointCloud.geometry.attributes.position.needsUpdate = true;
    edgeCloud.geometry.attributes.position.needsUpdate = true;
    /*
    entityGroup.children.forEach(element => {
        element.geometry.attributes.position.needsUpdate = true;
    });*/

    const time = Date.now() * 0.001;

	renderer.render( scene, camera );

    requestAnimationFrame( animate );
}




