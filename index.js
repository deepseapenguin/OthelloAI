import tf from '@tensorflow/tfjs-node';
import fs from 'fs';

const boardSize = 8;
const generateBoard = (size)=>{
    const lower = size/2-1

    const board = [];
    for(let i=0;i<lower;i++){
        const row=[];
        for(let j = 0;j<size;j++){
            row.push("-")
        }
        board.push(row)
    }

    const midRow1 = [];
    for(let i=0;i<lower;i++){
        midRow1.push("-")
    }
    midRow1.push("O")
    midRow1.push("X")
    for(let i=0;i<lower;i++){
        midRow1.push("-")
    }
    board.push(midRow1)

    const midRow2 = [];
    for(let i=0;i<lower;i++){
        midRow2.push("-")
    }
    midRow2.push("X")
    midRow2.push("O")
    for(let i=0;i<lower;i++){
        midRow2.push("-")
    }
    board.push(midRow2)

    for(let i=0;i<lower;i++){
        const row=[];
        for(let j = 0;j<size;j++){
            row.push("-")
        }
        board.push(row)
    }

    return board;
}

//const board = generateBoard(boardSize)
let board = [
    ["-","-","-","-","-","-","-","-",],
    ["-","-","-","-","-","-","-","-",],
    ["-","-","-","-","-","-","-","-",],
    ["-","-","-","O","X","-","-","-",],
    ["-","-","-","X","O","-","-","-",],
    ["-","-","-","-","-","-","-","-",],
    ["-","-","-","-","-","-","-","-",],
    ["-","-","-","-","-","-","-","-",],
]

const print = board=>{
    console.log("  0 1 2 3 4 5 6 7 ");
    for(let i in board){
        let str = i+" ";
        for(let j in board[i]){
            str+=board[i][j]+" "
        }

        console.log(str);
    }
    console.log("");
}

const printWithPossibleMove = (side,board) => {
    const moves = possibleMove(side,board)
    const nBoard = board.map(arr => arr.slice());
    const str = side.toLowerCase()
    console.log(side+" is moving");

    for (const k in moves) {
        const {i,j} = moves[k]

        nBoard[i][j] = "*";
    }


    console.log("  0 1 2 3 4 5 6 7 ");
    for(let i in nBoard){
        let str = i+" ";
        for(let j in nBoard[i]){
            str+=nBoard[i][j]+" "
        }

        console.log(str);
    }
    console.log("");
}

const direction = [
    {y:1,x:0},
    {y:-1,x:0},
    {y:0,x:1},
    {y:0,x:-1},
    {y:1,x:1},
    {y:1,x:-1},
    {y:-1,x:-1},
    {y:-1,x:1},
]

const possibleMove = (side, board)=>{
    const nBoard = board.map(arr => arr.slice());
    const possibleMove = []
    for(let i in nBoard){
        for(let j in nBoard[i]){
            if(nBoard[i][j]=="-"){
                if(checkMatch(i,j,side,nBoard)){
                    possibleMove.push({i,j})
                }
            }
        }
    }
    return possibleMove
}

const checkMatch = (i,j,side,board) => {
    let found = false
    let k = 0;
    while(k<8){
        found = checkDirection(i,j,direction[k],side,board)
        if(found){
            k=8
        }
        k++
    }

    return found
}

const checkDirection = (i,j,direction,side,board)=>{
    const opponent = side=="X"?"O":"X"
    let found = false

    let stop = false
    let posI = +i
    let posJ = +j
    const {y,x} = direction
    let opponentFound =  false
    while(!stop){
        posI+=y
        posJ+=x
        if(posI<=-1||posJ<=-1||posI>=boardSize||posJ>=boardSize){
            stop = true
        }else if(board[posI][posJ]=="-"){
            opponentFound=false
            stop=true
        }else if(board[posI][posJ]==opponent){
            opponentFound=true
        }else{
            stop=true
            found=opponentFound
        }
    }

    return found;
}

const flip = (i,j,direction,side,board)=>{
    let stop = false
    let posI = +i
    let posJ = +j
    const {y,x} = direction
    while(!stop){
        posI+=y
        posJ+=x
        if(board[posI][posJ] != side){
            board[posI][posJ] = side;
        }else{
            stop = true
        }
    }
}

const putPiece = (i,j,side,board) => {
    const nBoard = board.map(arr => arr.slice());
    nBoard[i][j] = side
    let k = 0;
    let found = false
    while(k<8){
        found = checkDirection(i,j,direction[k],side,nBoard)
        if(found){
            flip(i,j,direction[k],side,nBoard)
        }
        k++
    }
    return nBoard
}

const chooseMove = (side,board,model)=>{
    const possibleMoves = possibleMove(side, board)
    let highScore = -999999;
    let move = {}
    const opponent = side=="O"?"X":"O"
    
    for (const key in possibleMoves) {
        const {i,j} = possibleMoves[key]
        const nBoard = putPiece(i,j,side,board)
        const flatData = []

        for (const k in nBoard) {
            for (const l in nBoard[k]) {
                const data = nBoard[k][l]
                flatData.push(data=="-"?1:0)
                flatData.push(data==side?1:0)
                flatData.push(data==opponent?1:0)
            }
        }

        const score = Array.from(model.predict(tf.tensor(flatData,[1,boardSize*boardSize*3])).dataSync())[0];

        if(score>=highScore){
            move = {i,j}
            highScore = score
        }
    }

    return move
}

const createNNModel = () => {

    const model = tf.sequential({
        layers: [
          tf.layers.dense({inputShape: [boardSize*boardSize*3], units: boardSize*boardSize, activation: 'sigmoid'}),
          tf.layers.dense({units: 1, activation: 'sigmoid'}),
        ]
    });
    return model
}

const getBoardScore = (side,board)=>{
    //some magic
    return Math.random();
}

const movePiece = (side,board,model)=>{
    let nBoard = board.map(arr => arr.slice());
    const move = chooseMove(side,nBoard,model);
    if(move.i){
        nBoard = putPiece(move.i,move.j,side,nBoard)
        return nBoard
    }
}

const generateInitModels = num => {
    const models = []
    for (let i = 0; i < num; i++) {
        models.push({
            score:0,
            model:createNNModel()
        })
    }

    return models
}

const playMatch = (model1, model2) => {
    let board = generateBoard(8);
    let skip = false
    let end = false
    let side = "O"
    while(!end){
        const nBoard = movePiece(side,board, side=="O"?model1:model2)
        if(nBoard){
            board = nBoard
            //console.log(side+" turn");
            //print(board)
        }else{
            if(skip){
                end = true
            }else{
                skip = true
            }
        }

        side = side == "O" ? "X":"O"
    }
    return getWinner(board)

}

const playAllMatch = (models) =>{
    let maxI = models.length - 1
    let count = 0
    for (let i=0;i<maxI;i++) {
        let iniJ=i+1
        for(let j=iniJ;j<models.length;j++){
            const winner1 = playMatch(models[i].model, models[j].model)
            if(winner1=="O"){
                models[i].score = +models[i].score + 3
            }else if(winner1=="X"){
                models[j].score = +models[j].score + 3
            }else{
                models[i].score = +models[i].score + 1
                models[j].score = +models[j].score + 1
            }

            const winner2 = playMatch(models[j].model, models[i].model)
            if(winner2=="X"){
                models[i].score = +models[i].score + 3
            }else if(winner2=="O"){
                models[j].score = +models[j].score + 3
            }else{
                models[i].score = +models[i].score + 1
                models[j].score = +models[j].score + 1
            }
            count++
            console.log(count);
        }
    }
}

const getWinner = (board) => {
    let countO = 0
    let countX = 0
    for (const k in board) {
        for (const l in board[k]) {
            countO += board[k][l]=="O"?1:0
            countX += board[k][l]=="X"?1:0
        }
    }

    return countO>countX ? "O" : countX>countO ? "X" : "D"
}

const exportModel = async (generation,models)=>{
    const generationStr= ("000"+generation).slice(-4)
    if (!fs.existsSync("models/"+generationStr)){
        fs.mkdirSync("models/"+generationStr);
    }
    
    for(let i = 0;i<models.length;i++){
        const folderNum = ("000"+i).slice(-4)
        if (!fs.existsSync("models/"+generationStr+"/"+folderNum)){
            fs.mkdirSync("models/"+generationStr+"/"+folderNum);
        }
        await models[i].model.save("file:///HTML/OthelloAI/models/"+generationStr+"/"+folderNum);

    }
}

const generateAndExport = (num) =>{
    const models = generateInitModels(num);
    exportModel(0,models)
}



const importAll = async (generation,total)=>{
    let models = []
    for (let i = 0; i < total; i++) {
        const generationStr= ("000"+generation).slice(-4)
        const folderNum = ("000"+i).slice(-4)
        const model = await tf.loadLayersModel("file:///HTML/OthelloAI/models/"+generationStr+"/"+folderNum+"/model.json");
        models.push({
            score:0,
            model
        })
    }

    return models
}

const test1model = async (gen1,gen2,total)=>{
    let models = await importAll(gen1,total);
    let models2 = await importAll(gen2,total);
    let count = 0
    let score = 0;
    let j=total-1;

    for(let i = 0;i<total;i++){
        const winner1 = playMatch(models[i].model, models2[j].model)
        if(winner1=="O"){
        }else if(winner1=="X"){
            score+=3
        }else{
            score++
        }

        const winner2 = playMatch(models2[j].model, models[i].model)
        if(winner2=="X"){
        }else if(winner2=="O"){
            score+=3
        }else{
            score++
        }  
        console.log(++count);  
    }

    console.log("final score:"+score);
}

const test2model = async ()=>{
    let models1 = await importAll(0,50)[49];
    let models2 = await importAll(1,50)[49];

    const winner1 = playMatch(models1.model, models2.model)
    if(winner1=="O"){
    }else if(winner1=="X"){
        score++
    }else{
        score++
    }

    const winner2 = playMatch(models2[j].model, models[i].model)
    if(winner2=="X"){
    }else if(winner2=="O"){
        score++
    }else{
        score++
    }  
    console.log(++count);  
}

const test = async ()=>{
    let models = await importAll(1,50);

}

const population = models => {
    const newModels = []
    const take = 4
    for(let i = 1;i<=take;i++){
        newModels.push(models[models.length-i])
    }

    for(let i=0;i<take;i++){
        for(let j=0;j<4;j++){
            let rand=Math.floor(Math.random() * (take-1))
            if(rand==i){
                rand++
            }
            //console.log(rand);
            const model=crossoverAndMutation(newModels[i],newModels[rand])
            newModels.push(model)
        }
    }

    return newModels
}

const crossoverAndMutation = (model1, model2)=>{
    const weight11 = model1.model.layers[0].getWeights()[0].dataSync()
    const weight12 = model1.model.layers[1].getWeights()[0].dataSync()
    const weight21 = model2.model.layers[0].getWeights()[0].dataSync()
    const weight22 = model2.model.layers[1].getWeights()[0].dataSync()

    for(let i=0;i<weight11.length;i++){
        if(0.1 < Math.random()){
            weight11[i] = weight21[i]
        }

        if(0.01 < Math.random()){
            weight11[i] = weight11[i] * (1+Math.random())
        }
    }

    for(let i=0;i<weight12.length;i++){
        if(0.2 < Math.random()){
            weight12[i] = weight22[i]
        }

        if(0.05 < Math.random()){
            weight12[i] = weight12[i] * (1+Math.random())
        }
    }

    const model = createNNModel()
    model.layers[0].setWeights([tf.tensor(weight11,[192,64]), tf.zeros([64])])
    model.layers[1].setWeights([tf.tensor(weight12,[64,1]), tf.zeros([1])])

    return {
        model:model,
        score:0
    }
}



const main = async (num) => {
    let oldGen = num
    let oldModels = await importAll(oldGen,20)
    const models = population(oldModels)
    console.log(models.length)
    playAllMatch(models)
    for (const key in models) {
        console.log(key+":"+models[key].score);
    }
    models.sort((a,b)=>{
        if (a.score > b.score) return 1;
        if (b.score > a.score) return -1;

        return 0;
    })
    console.log("should be sorted");
    for (const key in models) {
        console.log(key+":"+models[key].score);
    }

    await exportModel(oldGen+1,models)
}

//generateAndExport(20)
test1model(0,11,20);
const fun = async ()=>{
    for(let i = 2;i<=10;i++){
        await main(i)
    }
}

//fun()

