import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import vertexShader from '../../shaders/vertex.glsl.js'
import fragmentShader from '../../shaders/fragment.glsl.js'

import {generateLegend} from '../legend/legendMaker.js';
import {makeNodes, makeConnectivityEdges, setNodePos, setEdgePosFromNodePos, computeClusterParams} from './graphMaker.js';
import {singleStepForceDirected, scaleToBounds} from '../force/force-directed.js'
import * as data from '../../saves/net_data_medium3.json' assert {type: 'json'}; // 63

console.log(data);

const fontPath = 'fonts/helvetiker_regular.typeface.json';

let camera, scene, renderer, stats;
let clusterGroup, clusMemberships, clusEdges;
let edgeConnectivity, edgePos;
let uiScene, orthoCamera;

// Legend Parameters
const defWidth = 900; 
const defHeight = 500; 

// Node & Edge Parameters
const sizeMult = .5;
const entityGeometry = new THREE.OctahedronGeometry( 0.05, 4 ); // 0.1, 4
const routerGeometry = new THREE.BoxGeometry(0.08, 0.08, 0.08); //0.08
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
    colorWithRisks: false,
    maxIter: 500,
    stepSize: .1,
    moveClusCenters: true,
    moveInterCluster: true,
    activateForce: false
};

// Read planar positions
const {pos, topologyEdges, risk_mean, risk_cov, funcEdges, entityColors, clusAssignments, extras} = data
const namesArr = Object.keys(pos);
const nNodes = namesArr.length;
const indDict = {}; // Dictionary of {name:index}
for (let i = 0; i < nNodes; i++) {
    indDict[namesArr[i]] = i;
}

let allNodePos = Array.from(Object.keys(pos), (key) => pos[key]); // Array of node positions sorted acc. to pos
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
        
        
        for ( let j = 0; j < clusterGroup.children.length; j++ ) {

            const cluster = clusterGroup.children[j];
            

            for ( let i = 0; i < cluster.children.length; i++ ) {

                const node = cluster.children[i];
                const name = node.name;

                if (value == true) {
                    node.material.color.setRGB( risk_mean[name] / extras.diam_z , 0, 0);
                } else {
                    node.material.color.setRGB(entityColors[name][0], entityColors[name][1], entityColors[name][2]);
                }
            
            }
        }

        
    } );

    gui.add( effectController, 'maxIter', 10, 100, 10).onChange( function ( value ){
        dt = stepSize / (value + 1);
    } );

    gui.add( effectController, 'stepSize', .05, .5, .05).onChange( function ( value ){
       stepSize=value;
    } );

    gui.add( effectController, 'moveClusCenters' );

    gui.add( effectController, 'moveInterCluster' );

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
    const planeGeometry = new THREE.PlaneGeometry( 16, 16 );
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

    clusterGroup = makeNodes(entityGeometry, routerGeometry, pos, funcEdges, risk_mean, entityColors,
         clusAssignments, extras, sizeMult, effectController.colorWithRisks); // Entity nodes and edges
    scene.add( clusterGroup );
    console.log(clusterGroup)

    // Edges
    // Connectivity

    edgeConnectivity = makeConnectivityEdges(edgeConnectivityMaterial, pos, funcEdges, risk_mean);
    
    scene.add( edgeConnectivity );
    edgeConnectivity.visible = effectController.showConnectivity;
    
    // Cluster parameters
    [clusMemberships, clusEdges] = computeClusterParams(clusterGroup, funcEdges, clusAssignments, indDict);
    console.log(clusEdges)
}


function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    // Leave Legend Same Size
    orthoCamera.left = -2 * window.innerWidth / defWidth + 1;
    orthoCamera.top = 1 * window.innerHeight / defHeight;
    orthoCamera.bottom = -1 * window.innerHeight / defHeight;
    orthoCamera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

}



//Mask the Array along axs given the boolean mask
function maskArray(array, indices, axs=0) {
    let tensor = tf.tensor(array);
    tensor = tf.gather(tensor, indices, axs);
    return tensor.arraySync();
}

function maskArray2(array, indices, axs=0) {
    
    const res = [];
    if (axs == 0) {
        indices.forEach( (i) => res.push(array[i]))
    } else if (axs == 1) {
        array.forEach( (row) => res.push( maskArray2(row, indices, 0) ) ) ;
    }
    return res;
}

// Move the nodes only within the cluster. TODO: Need to move to more robust datatype/table for sending data from py end
function moveWithinCluster(clusterGroup, allPosArr, allEdgeWeights, clusMemberships, stepSize=null, diamXY=1.3, minDist = 0.001, alpha=1){
    
    const nClus = clusterGroup.children.length;

    for (let j = 0; j < nClus; j++){
        const cluster = clusterGroup.children[j];
        
        // Compute masked pos and weights for jth cluster
        const jClusIndices = clusMemberships[j];
        let clusPosArr =  tf.tidy( () => maskArray2(allPosArr, jClusIndices));
        const clusEdgeWeights =  tf.tidy( () => maskArray2( maskArray2(allEdgeWeights, jClusIndices, 0), jClusIndices, 1) );        
        

        clusPosArr = calcMove(clusPosArr, clusEdgeWeights, stepSize, diamXY / nClus, false, minDist, alpha);
        setNodePos(cluster, clusPosArr);

        for (let k = 0; k < jClusIndices.length; k++){
            allPosArr[ jClusIndices[k]] = clusPosArr[k];
        }
        
    }
    
    return allPosArr;

}

// Move the center of the clusters wrt each other
function moveClusters(entityClusters, allPosArr, allEdgeWeights, clusAssignments, clusMemberships, intraClusEdges, stepSize=null, diamXY=2.3,minDist = 0.001, alpha=1){

    const nClus = entityClusters.children.length;
    // Form cluster center positions array
    let clusCenters = Array.from(entityClusters.children, (clus) => [clus.position.x, clus.position.y]);
    
    // Move the cluster centers
    clusCenters = calcMove( clusCenters, intraClusEdges, stepSize, diamXY, false, minDist, alpha);   
    
    const diff = new Array(nClus).fill(0).map(() => new Array(2).fill(0));
    for (let j = 0, clus; j < nClus; j++) {
        clus = entityClusters.children[j];
        diff[j] = [clusCenters[j][0] - clus.position.x, clusCenters[j][1] - clus.position.y];
        clus.position.set( clusCenters[j][0], clusCenters[j][1], 0);
    }

    // Update nodePos
    for ( let i = 0, clusIndex; i < nNodes; i ++ ) {
        clusIndex = clusAssignments[ Object.keys(pos)[i]];

        allPosArr[i][0] += diff[clusIndex][0];
        allPosArr[i][1] += diff[clusIndex][1];
    }

    return allPosArr
}


/**
 * Calculate the move of entities within a Group according to forces
 * @param {*} entityGroup Group of entities to be moves one step
 * @param {*} nodePos Array of node positions sorted according to the Group order
 * @param {*} edgeWeights Matrix of edge weights sorted wrt. the group order
 * @param {*} stepSize Expected step size of displacement
 * @param {*} diamXY Diameter of the node distribution (max - min)
 * @returns New array of node positions wrt. the group order
 */
function calcMove(nodePos, edgeWeights, stepSize=null, diamXY=1.3, scale=true, minDist = 0.001, alpha=1){
    
    let nodePosArray2d = tf.tidy( () =>  singleStepForceDirected(edgeWeights, nodePos, stepSize, diamXY, minDist, alpha));
    const bounds = {upper:[2, 2], lower:[-2, -2]};
    if ( scale) {
        nodePosArray2d = tf.tidy( () =>  scaleToBounds(nodePosArray2d, bounds) );
    }
    // TODO: Cooling and convergence criteria right here

    //console.log(tf.memory());

    return nodePosArray2d;
}


function animate() {
    
    const time = Date.now() * 0.001; 
    
    if (effectController.activateForce){
        if ( counter % nFrame == 0) {

            if (effectController.moveInterCluster) {
                allNodePos = moveWithinCluster(clusterGroup, allNodePos, funcEdges, clusMemberships, stepSize, 1.3, 0.001, .01); // 2.3
            }
            if (effectController.moveClusCenters) {
                allNodePos = moveClusters(clusterGroup, allNodePos, funcEdges, clusAssignments, clusMemberships, clusEdges, stepSize/2, 1.3)
            }
            setEdgePosFromNodePos(edgeConnectivity, allNodePos);
            //Topology edges?
            
            stepSize = (stepSize > dt) ? stepSize - dt : 0;
            
        }
        counter += 1;
    }

    renderer.clear();
	renderer.render( scene, camera );
    renderer.render( uiScene, orthoCamera );

    requestAnimationFrame( animate );

    stats.update();
}




