import { useState } from "react";
import { Box, List, ListItem, Paper, Typography } from "@mui/material";

function ItemDisplay({key, value}){  

  if(!value["data"] || value["data"] == undefined){
    return null;
  }

  return <Box sx={{width: "100%", display: "grid", gridTemplateColumns: "2fr 1fr"}} key={key}>
    {
      value["data"] &&
      value["type"] == "collectra.Image" || value["type"] == "collectra.ImageCrop" ?
      <img 
        style={{          
          width: "100%",                    
          display: "block",
          height: "auto",
        }}
        src={`data:image/jpg;base64,${value["data"]}`}
        alt={key}
      />
      :
      <Paper elevation={1} sx={{padding: "1rem", marginRight: "1rem", overflowY: "auto"}}>
        <Typography 
          variant="subtitle1"
        >          
          {String(value["data"]).slice(0, 1000)}{String(value["data"]).length > 1000 ? "..." : ""}
        </Typography>
      </Paper>
    }
    <Paper elevation={1} sx={{padding: "1rem", marginLeft: "1rem", overflowY: "auto"}}>
      {
        Object.entries(value).map(([subkey, subvalue]) => {
          if(subkey == "data"){
            return null;
          }
          return <Typography 
            key={subkey}              
            sx={{padding: "0.5rem"}}
            variant="subtitle2"
          >
            <strong>{subkey}:</strong>
            <br />
            {String(subvalue)}
          </Typography>
        })
      }
    </Paper>    
  </Box>
}

function FileDetail({ item }) {
  return (
    <Box sx={{
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
      gap: "1.5rem",
    }}>
      {item &&
        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          width: '100%',            
          rowGap: "1.5rem",      
        }}>                                     
          {Object.entries(item).map(([key, value]) => {
            if (value.constructor == Array){
              return value.map((subitem, index) => ItemDisplay({key: `${key}-${index}`, value: subitem}));
            }
            return ItemDisplay({key: key, value: value});
          })}                    
        </Box>
      }      
    </Box>
  )
}

function FileList({ items, onSelectItem }) {  
  return (
    <Box sx={{borderRight: '1px solid #ccc', paddingRight: '1rem'}}>
      <List>
        { items && 
          Object.entries(items).map(([key, item]) => (
            <ListItem button key={key} onClick={() => onSelectItem(key, item)}>
              {key}
            </ListItem>
          ))
        }        
      </List>
    </Box>
  );
}

export default function FileViewer({ items }) {

  const [selectedItem, setSelectedItem] = useState(null);

  const handleSelectItem = (key, item) => {
    setSelectedItem({
      key: key,
      ...item
    });
  };

  return (
    <Box
      sx={{
        width: '100%',
        display: 'grid',        
        gridTemplateColumns: '1fr 3fr',
        gap: '1rem',
      }}
    >
       <FileList items={items} onSelectItem={handleSelectItem} />       
       <FileDetail item={selectedItem} />
    </Box>
  );
}