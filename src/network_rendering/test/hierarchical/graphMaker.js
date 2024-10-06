import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';

export const color1 = new THREE.Color(44, 246, 4);
export const color2 = new THREE.Color(246, 4, 4);

// Makes and returns the entity Group
export function makeNodes(entityGeometry, routerGeometry, namesArr,  posArr, funcEdges, riskArr, entityColors, clusAssignment, extras, sizeMult=.5, colorWithRisks=true){
    
    const entityClustersGroup = new THREE.Group(); // Center of group is mean center of elements
    const nMembers = [];
    const clusCenters = [];    
    const entityIndexInClus = [];

    const nEntities = posArr.length;
    const nodeColors = new Float32Array( nEntities * 4 );
    const degrees = Array(nEntities);
    
    // Create Cluster Centers
    for ( let j = 0; j < extras.n_cluster; j++){
        entityClustersGroup.add( new THREE.Group());
        nMembers.push(0);
        clusCenters.push( new THREE.Vector3(0, 0, 0));
    }

    // Calculate nodeColors
    for ( let i = 0, clr, t, entityName; i < nEntities; i++ ) {
        entityName = namesArr[i];

        degrees[i] = funcEdges[i].reduce((acc, val) => acc + val );

        t = riskArr[i] / extras.diam_z > 0 ? riskArr[i] / extras.diam_z: 0;
        clr = colormapLinear(color1, color2, t);

        nodeColors[ i * 4 ] = colorWithRisks ? clr.r / 256 : entityColors[entityName][0];
        nodeColors[ i * 4 + 1] = colorWithRisks ? clr.g / 256 : entityColors[entityName][1];
        nodeColors[ i * 4 + 2] = colorWithRisks ? clr.b / 256 : entityColors[entityName][2];
        nodeColors[ i * 4 + 3] = colorWithRisks ? 1 : entityColors[entityName][3];

    }
    //const minDeg = Math.min(...degrees);
    //const maxDeg = Math.max(...degrees);
    //const nodeSizes = Array(nEntities);
    
    // Compute Cluster Centers & nMembers
    for ( let i = 0, entityName; i < nEntities; i ++ ) {
        entityName = namesArr[i];       
        nMembers[ clusAssignment[entityName] ] += 1;
        clusCenters[ clusAssignment[entityName] ].add( new THREE.Vector3(posArr[i][0], posArr[i][1], 0));    
    }

    // Center the Group 
    for ( let j=0; j < clusCenters.length; j++){
        clusCenters[j].divideScalar(nMembers[j]);
    }

    // Add entities to the entityClusterGroup
    for ( let i = 0, entityName, sizeScale, entitySampleMaterial, entity; i < nEntities; i ++ ){

        entityName = namesArr[i];       
        sizeScale = 1 //+ sizeMult * (degrees[i] - minDeg) / (maxDeg - minDeg);
        entitySampleMaterial = new THREE.MeshPhongMaterial({
            color:'#000000',
            emissive:'#000000',
            emissiveIntensity:1,
            specular:'#ffffff',
            shininess:30
        });

        entity = new THREE.Mesh( entityGeometry, entitySampleMaterial );
        //entity.scale.set(sizeScale, sizeScale, sizeScale);
        if (i % 3 == 0){
            entity.geometry = routerGeometry;
        }
        //entity.rotateX(Math.PI /2);

        let clusIndex = clusAssignment[entityName];
        entity.position.set(posArr[i][0], posArr[i][1], 0);
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

/**
 * Add new entities to the existing ones
 * 
 * @param {*} entityClustersGroup 
 * @param {*} newIdx Index that the new entities start
 * @param {*} nodeGeometry Geometry of the new entities
 * @param {*} namesArr Name array of all entities
 * @param {*} posArr Position array of all entities
 * @param {*} riskArr Risk array of all entities
 * @param {*} diamXY Diameter in X and Y direction (assumed identical)
 * @param {*} diamZ Diameter in Z direction
 */
export function addNodesSimple(entityClustersGroup, clusMemberships, clusAssignments, entityIndexInClus, newIdx, nodeGeometry, namesArr, posArr, riskArr, diamXY, diamZ){
    
    const nEntities = namesArr.length;
    const nodeColors = new Float32Array( (nEntities - newIdx) * 4 );
    
    // Compute Node Colors
    for ( let k = 0, clr, t, entityName; k < nEntities - newIdx; k++ ) {
        entityName = namesArr[k + newIdx];

        t = riskArr[k + newIdx] > 0 ? riskArr[k + newIdx] / diamZ: 0;
        clr = colormapLinear(color1, color2, t);

        nodeColors[ k * 4 ] = clr.r / 256;
        nodeColors[ k * 4 + 1] = clr.g / 256;
        nodeColors[ k * 4 + 2] = clr.b / 256;
        nodeColors[ k * 4 + 3] = 1;

    }

    // Compute Positions
    for ( let i = newIdx, entityName; i < nEntities; i ++ ) {
        entityName = namesArr[i];       
        posArr.push([Math.random() * diamXY - diamXY / 2, Math.random() * diamXY - diamXY / 2]);
    }


    // Add entities to the entityClusterGroup
    for ( let i = newIdx, k, entitySampleMaterial, entity; i < nEntities; i ++ ){
        k = i - newIdx;     
        entitySampleMaterial = new THREE.MeshPhongMaterial({
            color:'#000000',
            emissive:'#000000',
            emissiveIntensity:1,
            specular:'#ffffff',
            shininess:30
        });        

        entity = new THREE.Mesh( nodeGeometry, entitySampleMaterial );

        let clusIndex = 0;
        entity.position.set(posArr[i][0], posArr[i][1], 0);
        entity.material.color.setRGB(nodeColors[ 4 * k ], nodeColors[ 4 * k + 1], nodeColors[ 4 * k + 2]);
        entity.name = namesArr[i];

        entityClustersGroup.children[ clusIndex].add( entity );
        clusMemberships[clusIndex].push(i);
        clusAssignments[entity.name] = clusIndex;
        entityIndexInClus[entity.name] = i;
    }
}

/**
 * Activate the already existing nodes in entityClustersGroup. Used when new entities are added to network of interest
 * @param {*} entityClustersGroup 
 * @param {*} newIdx Index that the new entities start
 * @param {*} nodeGeometry Geometry of the new entities
 * @param {*} namesArr Name array of all entities
 * @param {*} posArr Position array of all entities
 * @param {*} riskArr Risk array of all entities
 * @param {*} diamXY Diameter in X and Y direction (assumed identical)
 * @param {*} diamZ Diameter in Z direction
 */
export function activateNodes(entityClustersGroup, clusMemberships, clusAssignments, entityIndexInClus, indDict, activeNodes, newIdx, nodeGeometry, namesArr, posArr, riskArr, diamXY, diamZ) {

    const nEntities = namesArr.length;
    const nodeColors = new Float32Array( (nEntities - newIdx) * 4 );
    
    // Compute Node Colors
    for ( let k = 0, clr, t, entityName; k < nEntities - newIdx; k++ ) {
        entityName = namesArr[k + newIdx];

        t = riskArr[k + newIdx] > 0 ? riskArr[k + newIdx] / diamZ: 0;
        clr = colormapLinear(color1, color2, t);

        nodeColors[ k * 4 ] = clr.r / 256;
        nodeColors[ k * 4 + 1] = clr.g / 256;
        nodeColors[ k * 4 + 2] = clr.b / 256;
        nodeColors[ k * 4 + 3] = 1;

    }

    // Edit Positions
    for ( let i = newIdx, entityName; i < nEntities; i ++ ) {
        entityName = namesArr[i];       
        posArr[i] = [Math.random() * diamXY - diamXY / 2, Math.random() * diamXY - diamXY / 2];
    }

    // Add entities to the entityClusterGroup
    for ( let i = newIdx, k, sizeScale, entitySampleMaterial, entity; i < nEntities; i ++ ){
        k = i - newIdx;

        let clusIndex = 0;
        entity = entityClustersGroup.children[ clusIndex].children[i]; 
        //entity.geometry = nodeGeometry;
        entity.position.set(posArr[i][0], posArr[i][1], 0);
        entity.material.color.setRGB(nodeColors[ 4 * k ], nodeColors[ 4 * k + 1], nodeColors[ 4 * k + 2]);
        entity.name = namesArr[i];
        entity.visible = true;

        clusMemberships[clusIndex][i] = i;
        clusAssignments[entity.name] = clusIndex;
        entityIndexInClus[entity.name] = i;
        indDict[entity.name] = i;
        activeNodes[clusIndex].push(i);
    }
}

// Makes and returns the connectivity edges mesh object
export function makeConnectivityEdges(edgeConnectivityMaterial, nodePosArr, edgesArr){

    const edgePos = nodePos2AllEdgePos(nodePosArr)
    const edgeColors = nodePos2EdgeColor(edgesArr);

    const edgeConnectivityGeometry = new THREE.BufferGeometry();
    edgeConnectivityGeometry.setAttribute( 'position', new THREE.BufferAttribute( edgePos, 3 ) );
    edgeConnectivityGeometry.setAttribute( 'color', new THREE.Uint8BufferAttribute( edgeColors, 4, true ) );
    
    const edgeConnectivity = new THREE.LineSegments( edgeConnectivityGeometry, edgeConnectivityMaterial );
    
    return edgeConnectivity
}

// Makes and returns the topology edges mesh object
export function makeTopologyEdges(edgeTopologyMaterial, nodePosArr, edgeList, indDict){

    const edgePositions = nodePos2EdgePos(nodePosArr, edgeList, indDict)

    const edgeTopologyGeometry = new THREE.BufferGeometry();
    edgeTopologyGeometry.setAttribute( 'position', new THREE.BufferAttribute( edgePositions, 3 ) );
    
    return new THREE.LineSegments( edgeTopologyGeometry, edgeTopologyMaterial );    
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

// Returns edge colors buffer from node position array
function nodePos2EdgeColor(edgesArr){
    const nNodes = edgesArr.length;
    const edgeColor = new Float32Array( 4 * 2 * nNodes * (nNodes - 1) );

    for (let i = 0; i < nNodes ; i++) {

        for (let j = 0, k; j < nNodes ; j++) {

            if (j == i){
                continue;
            }

            k = i * nNodes + j;

            edgeColor[ 8 * k ] = (edgesArr[i][j])** (1/3) * 255 ; 
            edgeColor[ 8 * k + 1] = 0;
            edgeColor[ 8 * k + 2] = 0;
            edgeColor[ 8 * k + 3] = (edgesArr[i][j]) ** (3) * 255;

            edgeColor[ 8 * k + 4] = edgeColor[ 8 * k ]; 
            edgeColor[ 8 * k + 5] = edgeColor[ 8 * k + 1];
            edgeColor[ 8 * k + 6] = edgeColor[ 8 * k + 2];
            edgeColor[ 8 * k + 7] = edgeColor[ 8 * k + 3];
        }
    }
    return edgeColor
}

// Returns edge position buffer from node position array
function nodePos2EdgePos(nodePosArr, edgeList, indDict){

    const nEdges = edgeList.length;
    const edgePositions = new Float32Array( nEdges * 2 * 3 );
    if (nEdges == 0){
        return edgePositions;
    }

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
