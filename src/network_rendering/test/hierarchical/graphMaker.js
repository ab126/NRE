import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';

export const color1 = new THREE.Color(44, 246, 4);
export const color2 = new THREE.Color(246, 4, 4);

// Makes and returns the entity Group
export function makeNodes(entityGeometry, routerGeometry,  pos, funcEdges, risk_mean, entityColors, clusAssignment, extras, sizeMult=.5, colorWithRisks=true){
    
    const entityClustersGroup = new THREE.Group(); // Center of group is mean center of elements
    const nMembers = [];
    const clusCenters = [];    
    const entityIndexInClus = [];

    const nEntities = Object.keys(pos).length;
    const nodeColors = new Float32Array( nEntities * 4 );
    const nodePosArray = Array(nEntities);
    const degrees = Array(nEntities);
    
    for ( let j = 0; j < extras.n_cluster; j++){
        entityClustersGroup.add( new THREE.Group());
        nMembers.push(0);
        clusCenters.push( new THREE.Vector3(0, 0, 0));
    }

    for ( let i = 0, clr, t, entityName; i < nEntities; i++ ) {
        entityName = Object.keys(pos)[i];

        nodePosArray[i] = pos[entityName];
        degrees[i] = funcEdges[i].reduce((acc, val) => acc + val );

        t = risk_mean[entityName] / extras.diam_z > 0 ? risk_mean[entityName] / extras.diam_z: 0;
        clr = colormapLinear(color1, color2, t);

        nodeColors[ i * 4 ] = colorWithRisks ? clr.r / 256 : entityColors[entityName][0];
        nodeColors[ i * 4 + 1] = colorWithRisks ? clr.g / 256 : entityColors[entityName][1];
        nodeColors[ i * 4 + 2] = colorWithRisks ? clr.b / 256 : entityColors[entityName][2];
        nodeColors[ i * 4 + 3] = colorWithRisks ? 1 : entityColors[entityName][3];

    }
    const minDeg = Math.min(...degrees);
    const maxDeg = Math.max(...degrees);
    const nodeSizes = Array(nEntities);

    
    
    // Compute Cluster Centers & nMembers
    for ( let i = 0, entityName; i < nEntities; i ++ ) {
        entityName = Object.keys(pos)[i];         
        nMembers[ clusAssignment[entityName] ] += 1;
        clusCenters[ clusAssignment[entityName] ].add( new THREE.Vector3(nodePosArray[i][0], nodePosArray[i][1], 0));    
    }


    // Center the Group 
    for ( let j=0; j < clusCenters.length; j++){
        clusCenters[j].divideScalar(nMembers[j]);
    }

    // Add entities to the entityClusterGroup
    for ( let i = 0, entityName, sizeScale, entitySampleMaterial, entity; i < nEntities; i ++ ){

        entityName = Object.keys(pos)[i];      
        sizeScale = 1 + sizeMult * (degrees[i] - minDeg) / (maxDeg - minDeg);
        entitySampleMaterial = new THREE.MeshPhongMaterial({
            color:'#000000',
            emissive:'#000000',
            emissiveIntensity:1,
            specular:'#ffffff',
            shininess:30
        });

        entity = new THREE.Mesh( entityGeometry, entitySampleMaterial );
        entity.scale.set(sizeScale, sizeScale, sizeScale);
        if (i % 3 == 0){
            entity.geometry = routerGeometry;
        }
        //entity.rotateX(Math.PI /2);

        let clusIndex = clusAssignment[entityName];
        entity.position.set(nodePosArray[i][0], nodePosArray[i][1], 0);
        entity.position.add( clusCenters[ clusIndex].clone().negate()  );
        entity.material.color.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2]);
        entity.name = entityName;

        entityIndexInClus.push( entityClustersGroup.children[ clusIndex].children.length);
        entityClustersGroup.children[ clusIndex].add( entity );
    }

    // Move the center of the cluster to its original position
    for (let j=0; j < clusCenters.length; j++) {
        entityClustersGroup.children[j].position.add( clusCenters[j]);
    }

    return [entityClustersGroup, entityIndexInClus];
}

// Makes and returns the connectivity edges mesh object
export function makeConnectivityEdges(edgeConnectivityMaterial, pos, funcEdges, risk_mean){

    const [edgePos, edgeColors] = makeAllEdges( pos, funcEdges, risk_mean, false);

    const edgeConnectivityGeometry = new THREE.BufferGeometry();
    edgeConnectivityGeometry.setAttribute( 'position', new THREE.BufferAttribute( edgePos, 3 ) );
    edgeConnectivityGeometry.setAttribute( 'color', new THREE.Uint8BufferAttribute( edgeColors, 4, true ) );
    
    const edgeConnectivity = new THREE.LineSegments( edgeConnectivityGeometry, edgeConnectivityMaterial );
    
    return edgeConnectivity
    
}

// TODO: Might not need makers using pos
/**
 * Makes edge defining points from functional connectivity edge array
 * @param {*} pos Object whose fields are entity ids with [pos_x, pos_y] array describing 2D position
 * @param {*} edges Weight Matrix of normalized edge weights
 * @param {*} elavate If true, elavates the entities according the mean value of their risks in z direction
 * @returns [edgeConnectivityPositions, edgeColors]
 *  edgeConnectivityPositions: Float32Array of source position coordinates and destination  position coordinates
 *  edgeColors: Float32Array of RGBA values of edges
 */
export function makeAllEdges(pos, edges, risk, elavate) {
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

// Makes and returns the topology edges mesh object
export function makeTopologyEdges(edgeTopologyMaterial, pos, edgeList){

    const edgePositions = makeEdgePositions(edgeList, pos, false);

    const edgeTopologyGeometry = new THREE.BufferGeometry();
    edgeTopologyGeometry.setAttribute( 'position', new THREE.BufferAttribute( edgePositions, 3 ) );
    
    return new THREE.LineSegments( edgeTopologyGeometry, edgeTopologyMaterial );
    
}

/**
 * Makes edge defining points
 * @param {*} src Source entity
 * @param {*} dst Destination entity
 */
function makeEdgePositions(edgeList, pos, elavate){
    const nEdges = edgeList.length
    const edgePositions = new Float32Array( nEdges * 2 * 3 );
    let src, dst;

    for (let i = 0; i < nEdges; i++) {

        [src, dst] = edgeList[i];

        edgePositions[ 6 * i ] = pos[src][0]; 
        edgePositions[ 6 * i + 1] = pos[src][1];
        edgePositions[ 6 * i + 2] = elavate ? risk_mean[src] : 0;

        edgePositions[ 6 * i + 3] = pos[dst][0]; 
        edgePositions[ 6 * i + 4] = pos[dst][1];
        edgePositions[ 6 * i + 5] = elavate ? risk_mean[dst] : 0;

    }
    return edgePositions
}

// For altering Positions

// Sets the node positions according to the new node position array
export function setNodePos(nodeGroup, nodePos){
    const nNodes = nodeGroup.children.length;
    const center = nodeGroup.position.clone();
    for ( let i = 0; i < nNodes; i ++ ) {

        const node = nodeGroup.children[i];
        node.position.set(nodePos[i][0] - center.x, nodePos[i][1] - center.y, 0);
    }
}

// Get all possible edge positions from node positions
export function setAllEdgePosFromNodePos(edgesObject, nodePos) {
    const edgePos = nodePos2AllEdgePos(nodePos);
    edgesObject.geometry.setAttribute( 'position', new THREE.BufferAttribute( edgePos, 3 ) );
    edgesObject.geometry.attributes.position.needsUpdate = true;
}

// Get the edge positions from node positions
export function setEdgePosFromNodePos(edgesObject, nodePos, edgeList, indDict) {
    const edgePos = nodePos2EdgePos(nodePos, edgeList, indDict);
    edgesObject.geometry.setAttribute( 'position', new THREE.BufferAttribute( edgePos, 3 ) );
    edgesObject.geometry.attributes.position.needsUpdate = true;
}

// Returns edge position buffer from node position array
function nodePos2AllEdgePos(nodePosArr){
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

// Returns edge position buffer from node position array
function nodePos2EdgePos(nodePosArr, edgeList, indDict){

    const nEdges = edgeList.length
    const edgePositions = new Float32Array( nEdges * 2 * 3 );

    for (let k = 0, i, j, src, dst; k < nEdges; k++) {

        [src, dst] = edgeList[k];
        i = indDict[src];
        j = indDict[dst];

        edgePositions[ 6 * k ] = nodePosArr[i][0];
        edgePositions[ 6 * k + 1] = nodePosArr[i][1];
        edgePositions[ 6 * k + 2] = 0;

        edgePositions[ 6 * k + 3] = nodePosArr[j][0]; 
        edgePositions[ 6 * k + 4] = nodePosArr[j][1];
        edgePositions[ 6 * k + 5] = 0;

    }
    return edgePositions;
}


// Compute the Cluster Related parameters ahead of time to reduce overhead
export function computeClusterParams(clusterGroup, allEdgeWeights, clusAssignments, indDict){

    const nNodes = clusAssignments.length;
    const nClus = clusterGroup.children.length;
    const clusMemberships = [];
    
    for (let j = 0; j < nClus; j++){
        const cluster = clusterGroup.children[j];
        
        // Form Mask
        const jClusIndices = []; // Indices of the entities that belong to cluster j
        for (let k = 0; k < cluster.children.length; k++){

            const name = cluster.children[k].name;
            if (clusAssignments[name] == j){
                jClusIndices.push(indDict[name]);
            } 
        }
        clusMemberships.push(jClusIndices);
        
    }

    // Calculated weighted mass divided edge weights
    const clusEdges = new Array(nClus).fill(0).map(() => new Array(nClus).fill(0));
    for (let i = 0, src_clus, dst_clus, mass; i < nNodes; i ++) {

        for (let j = 0; j < nNodes; j ++ ) {
            if (i == j) {
                continue
            }

            src_clus = clusAssignments[Object.keys(pos)[i]];
            dst_clus = clusAssignments[Object.keys(pos)[j]];
            mass = clusterGroup.children[dst_clus].children.length;

            clusEdges[src_clus][dst_clus] += allEdgeWeights[i][j] / mass;
        }
    }

    return [clusMemberships, clusEdges];

}


/**
 * Linear Interpolation between color1 and color2
 * @param {*} color1 THREE.Color :  Start color
 * @param {*} color2 THREE.Color :  End color
 * @param {*} t float : Interpolation parameter in [0, 1]
 */
export function colormapLinear(color1, color2, t){

    const p3 =  tf.tidy( () => colormapHelper(color1, color2, t));
    return new THREE.Color(parseInt(p3[0]), parseInt(p3[1]), parseInt(p3[2]));

    function colormapHelper(color1, color2, t){
        const p1 = tf.tensor( [color1.r, color1.g, color1.b]);
        const p2 = tf.tensor( [color2.r, color2.g, color2.b]);

        return p1.add( p2.sub(p1).mul(t)).arraySync();
    }
    
}
