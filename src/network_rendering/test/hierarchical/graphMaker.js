import * as THREE from 'three';

// Makes and returns the entity Group
export function makeNodes(entityGeometry, routerGeometry,  pos, funcEdges, risk_mean, entityColors, clusAssignment, extras, sizeMult=.5, colorWithRisks=true){
    
    const entityClustersGroup = new THREE.Group(); // Center of group is mean center of elements
    const nMembers = [];
    const clusCenters = [];

    const nEntities = Object.keys(pos).length;
    const nodeColors = new Float32Array( nEntities * 4 );
    const nodePosArray = Array(nEntities);
    const degrees = Array(nEntities);
    
    for ( let j = 0; j < extras.n_cluster; j++){
        entityClustersGroup.add( new THREE.Group());
        nMembers.push(0);
        clusCenters.push( new THREE.Vector3(0, 0, 0));
    }

    for ( let i = 0, entityName; i < nEntities; i++ ) {
        entityName = Object.keys(pos)[i];

        nodePosArray[i] = pos[entityName];
        degrees[i] = funcEdges[i].reduce((acc, val) => acc + val );

        nodeColors[ i * 4 ] = colorWithRisks ? risk_mean[entityName] / extras.diam_z : entityColors[entityName][0];
        nodeColors[ i * 4 + 1] = colorWithRisks ? 0 : entityColors[entityName][1];
        nodeColors[ i * 4 + 2] = colorWithRisks ? 0 : entityColors[entityName][2];
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
        entity.rotateX(Math.PI /2);

        let clusIndex = clusAssignment[entityName];
        entity.position.set(nodePosArray[i][0], nodePosArray[i][1], 0);
        entity.position.add( clusCenters[ clusIndex].clone().negate()  );
        entity.material.color.setRGB(nodeColors[ 4 * i ], nodeColors[ 4 * i + 1], nodeColors[ 4 * i + 2]);
        entity.name = entityName;

        entityClustersGroup.children[ clusIndex].add( entity );

    }

    // Move the center of the cluster to its original position
    for (let j=0; j < clusCenters.length; j++) {
        entityClustersGroup.children[j].position.add( clusCenters[j]);
    }

    return entityClustersGroup;
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

/**
 * Makes edge defining points from functional connectivity edge array
 * @param {*} pos Object whose fields are entity ids with [pos_x, pos_y] array describing 2D position
 * @param {*} edges Weight Matrix of normalized edge weights
 * @param {*} elavate If true, elavates the entities according the mean value of their risks in z direction
 * @returns [edgeConnectivityPositions, edgeColors]
 *  edgeConnectivityPositions: Float32Array of source position coordinates and destination  position coordinates
 *  edgeColors: Float32Array of RGBA values of edges
 */
function makeAllEdges(pos, edges, risk, elavate) {
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



