import * as THREE from 'three';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';

// Legend Parameters
const defWidth = 900; 
const defHeight = 500; 

export class fontManager{
    constructor(fontPath){
        this.path = fontPath;
    }

    addFont(message, textPos, textMaterial, group2Add=null, size=0.03, scale=null){

        const loader = new FontLoader();
        let text;
    
        loader.load( this.path, function ( font ) {
    
            const shapes = font.generateShapes(message, size);
            const geometry = new THREE.ShapeGeometry( shapes );
            text = new THREE.Mesh( geometry, textMaterial );
            if (scale == null) {
                text.scale.set(0.6, 1, 1);
            } else {
                text.scale.set(...scale);
            }         
            text.position.set(...textPos); 
            if (group2Add != null) {
                group2Add.add( text );
            }
    
            //animate();
    
        } );
    }
}


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
    const xPos = Array( -widthPerc * legendWidth * 0.5, -widthPerc * legendWidth * 0.35, -widthPerc * legendWidth * 0.2);

    return [xPos, yPos]
}

export function generateLegend(fontPath, entityGeometry, routerGeometry, nodeMaterial, connectivityMaterial, topologyMaterial){

    // Scene
    const uiScene = new THREE.Scene();
    const orthoCamera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, .1, 2 );
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
    const [legendWidth, legendHeight] = [0.4, 1.5];
    const sprite = new THREE.Sprite( new THREE.SpriteMaterial( { color:'#424242' } ) );
    sprite.scale.set(0.4, 1.5, 1);
    sprite.position.set(0, 0, 0);
    legend.add( sprite );

    // Text Parameters
    const fm = new fontManager(fontPath);
    const textYShift = 0.02;

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
    const [xPos0, yPos0] = [-.05, legendHeight/2 - 3*textYShift];
    const margin = 0.01;

    fm.addFont("Legend", [xPos0, yPos0 , 0], headerMat, legendHeader);
    
    const pts = new Float32Array([-legendWidth/2 + margin, yPos0 - textYShift - margin, 0,  legendWidth/2 - margin, yPos0 - textYShift - margin, 0 ]); // -.19, .39, 0,  .17, .39, 0 
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute( 'position', new THREE.BufferAttribute( pts, 3 ).setUsage( THREE.DynamicDrawUsage ) );
    const line = new THREE.Line(lineGeometry, headerMat);
    legendHeader.add(line);

    // Node Segment
    const [xPos1, yPos1] = legendSegmentLocations(3, 0.5, 0.35);

    fm.addFont("Entity", [ xPos1[0], yPos1[0] - textYShift, 0], liteMat, nodeSegment); // -.09, .22
    
    const routerSample = new THREE.Mesh( routerGeometry, nodeMaterial );
    routerSample.position.set(xPos1[1], yPos1[1], 0); // -0.13, 0.25
    routerSample.scale.set(0.25, 0.5, 0.5);
    nodeSegment.add( routerSample );
    fm.addFont(": Router/Switch", [ xPos1[2], yPos1[1] - textYShift, 0], liteMat, nodeSegment); // -.09, .22
    
    const entitySample = new THREE.Mesh( entityGeometry, nodeMaterial );
    entitySample.position.set(xPos1[1], yPos1[2], 0); // -0.13, 0.25
    entitySample.scale.set(0.25, 0.5, 0.5);
    nodeSegment.add( entitySample );
    fm.addFont(": Endpoints", [ xPos1[2], yPos1[2] - textYShift, 0], liteMat, nodeSegment); // -.09, .22


    // Connectivity Segment

    const [ptsShiftx, ptsShifty] = [0.025, 0.05];
    const [xPos2, yPos2] = legendSegmentLocations(3, 0.15, 0.4);
    

    const edgeSample = new THREE.Group();
    connectivitySegment.add(edgeSample);

    fm.addFont("Connectivity", [xPos2[0], yPos2[0] - textYShift, 0], liteMat, connectivitySegment); //  -.09, .04, 0

    const edgePointPositions = new Float32Array([xPos2[1] - ptsShiftx, yPos2[1] - ptsShifty, .01, xPos2[1] + ptsShiftx, yPos2[1] + ptsShifty, .01 ]); //[-.16, .0, .1, -.11, .10, .1 ]
    const edgeGeometry = new THREE.BufferGeometry()
    edgeGeometry.setAttribute( 'position', new THREE.BufferAttribute( edgePointPositions, 3 ) );    
    const edge = new THREE.LineSegments( edgeGeometry, connectivityMaterial );
    edgeSample.add(edge);
    fm.addFont(": Functional\n Connectivity", [xPos2[2], yPos2[1] , 0], liteMat, connectivitySegment); //  -.09, .04, 0

    const edgePointPositions2 = new Float32Array([xPos2[1] - ptsShiftx, yPos2[2] - ptsShifty, .01, xPos2[1] + ptsShiftx, yPos2[2] + ptsShifty, .01 ]); //[-.16, .0, .1, -.11, .10, .1 ]
    const edgeGeometry2 = new THREE.BufferGeometry()
    edgeGeometry2.setAttribute( 'position', new THREE.BufferAttribute( edgePointPositions2, 3 ) );    
    const edge2 = new THREE.LineSegments( edgeGeometry2, topologyMaterial );
    edgeSample.add(edge2);
    fm.addFont(": Topology", [xPos2[2], yPos2[2] - textYShift, 0], liteMat, connectivitySegment); //  -.09, .04, 0


    // Risk Segment
    const [xPos3, yPos3] = legendSegmentLocations(4, -0.2, 0.3);

    const colorNode = new THREE.Mesh( entityGeometry, nodeMaterial );
    colorNode.position.set(-0.13, -.14, 0);
    colorNode.scale.set(0.5, 0.5, 0.5);
    //riskSegment.add(colorNode);

    fm.addFont("Risk", [ xPos3[0], yPos3[0] - textYShift, 0], liteMat, riskSegment); // -.09, -.14, 0
    fm.addFont(": Prior", [ xPos3[2], yPos3[1] - textYShift, 0], liteMat, riskSegment); 
    fm.addFont(": Measured", [ xPos3[2], yPos3[2] - textYShift, 0], liteMat, riskSegment); 
    fm.addFont(": Estimated", [ xPos3[2], yPos3[3] - textYShift, 0], liteMat, riskSegment); 

    // Nomenclature
    const [xPos4, yPos4] = legendSegmentLocations(7, -0.55, 0.38);
    //const noteSize = 0.03 * 2.5/3;

    fm.addFont("Nomenclature", [ xPos4[0], yPos4[0] - textYShift, 0], liteMat, riskSegment); // -.09, -.14, 0
    fm.addFont("   [0,1]: 0 - low risk, 1 - high risk ", [ xPos4[0], yPos4[1] - textYShift, 0], liteMat, riskSegment, 0.025); 
    fm.addFont("   Transparancy: Strenght of Connectivity ", [ xPos4[0], yPos4[2] - textYShift, 0], liteMat, riskSegment, 0.025); 
    fm.addFont("   Color: green - low, red - high ", [ xPos4[0], yPos4[3] - textYShift, 0], liteMat, riskSegment, 0.025); 
    fm.addFont("   Circle: Entity ", [ xPos4[0], yPos4[4] - textYShift, 0], liteMat, riskSegment, 0.025); 
    fm.addFont("   Arc: Connectivity ", [ xPos4[0], yPos4[5] - textYShift, 0], liteMat, riskSegment, 0.025); 
    fm.addFont("   Circle Size: Traffic Quantity", [ xPos4[0], yPos4[6] - textYShift, 0], liteMat, riskSegment, 0.025); 
    
    //console.log(uiScene)
    return [uiScene, orthoCamera]

}
