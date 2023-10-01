import fs from "fs";
import path from "path";

(async () => {
    const dataDir = path.resolve(__dirname, "Data", "vlsp2020_train_set_02")
    const dataArr = fs.readdirSync(dataDir)
    const txtArr = dataArr.filter(file => file.toLowerCase().endsWith(".txt"))
    const audioArr = dataArr.filter(file => file.toLowerCase().endsWith(".wav"))
    console.log("Total Data Files Found: ", dataArr.length)
    console.log("Total Text Files Found: ", txtArr.length)
    console.log("Total Audio Files Found: ", audioArr.length)
    console.log("Total Valid Files Found: ", audioArr.length + txtArr.length, "\n")

    let csvContent = ""
    const newDir = path.resolve(__dirname, "datasets")
    if(!fs.existsSync(newDir)) fs.mkdirSync(newDir)

    for(const file of audioArr) {
        csvContent += file.split(".")[0] 
                    + "|" + fs.readFileSync(path.join(dataDir, txtArr[audioArr.indexOf(file)]), "utf-8")
                    + "|" + fs.readFileSync(path.join(dataDir, txtArr[audioArr.indexOf(file)]), "utf-8")
                    + "\r\n"
        fs.copyFileSync(path.join(dataDir, file), path.join(newDir, file))
    }

    fs.writeFileSync(path.join(newDir, "metadata_VI.csv"), csvContent)
    // console.log(fs.readFileSync("./dataset/metadata.csv", "utf-8"))
})()
