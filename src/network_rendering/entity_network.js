import * as THREE from 'three';

import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import * as data from './saves/net_data2.json' assert {type: 'json'}; 

let camera, scene, renderer, controls;
let entityGroup;
let nodeGeometry, nodePositions, nodeCloud;
let edgeGeometry, edgePositions, edgeCloud;
const radius = 0.05;
const dz = 0.01

const effectController = {
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
        /*
        Object.keys(pos).forEach((name, index) => {
            dodec = nodes[index]
            changeElavation(name, dodec, value)
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
    
    entityGroup = new THREE.Group(); // Entity nodes and edges
    scene.add( entityGroup );

    // Nodes
    
    nodeGeometry = new THREE.BufferGeometry();
    nodePositions = new Float32Array( nNodes * 3 );

    //const nodeMaterial = new THREE.MeshStandardMaterial( {color: 0x242323} );
    const nodeMaterial = new THREE.PointsMaterial( {
        color: 0xFFFFFF,
        size: 6,
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
        
    nodeCloud = new THREE.Points( nodeGeometry, nodeMaterial );
    entityGroup.add(nodeCloud)
    
    /*
    const nodes = [];
    Object.keys(pos).forEach(element => {
        nodes.push(makeNode(element, nodeMaterial, radius))
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
    entityGroup.add( edgeCloud );

    const controls = new OrbitControls( camera, renderer.domElement );

    /**
     * Makes and adds entity as nodes to the scene
     * @param {*} name 
     * @param {*} material
     * @param {*} radius Node raidus
     */
    function makeNode(name, material, radius = 0.05) {

        const geometry_dodec = new THREE.DodecahedronGeometry( radius );
        const dodec = new THREE.Mesh( geometry_dodec, material );
        dodec.position.x = pos[name][0]
        dodec.position.y = pos[name][1]
        dodec.position.z = 0 //risk_mean[name]
        
        scene.add( dodec );

        return dodec
    }

    /**
     * Elavataes the nodes according to risk levels
     * @param {*} name 
     * @param {*} dodec 
     * @param {*} elavate If true elavate according to risks 
     * @returns 
     */
    function changeElavation(name, dodec, elavate) {

        dodec.position.z = elavate ? risk_mean[name] : 0
        
        scene.add( dodec );
    }

    
    
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
    
    nodeCloud.geometry.attributes.position.needsUpdate = true;
    edgeCloud.geometry.attributes.position.needsUpdate = true;

    const time = Date.now() * 0.001;
    
    //cube.rotation.y = time * 0.4;
	
	//controls.update();

	renderer.render( scene, camera );

    requestAnimationFrame( animate );
}




