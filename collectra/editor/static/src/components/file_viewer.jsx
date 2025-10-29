import { useState } from "react";
import { Box, List, ListItem, Paper } from "@mui/material";

function ItemDisplay({key, value}){  

  if(!value["data"] || value["data"] == undefined){
    return null;
  }

  return <Box sx={{width: "100%"}} key={key}>
    {
      value["data"] &&
      <img 
        sx={{
          maxWidth: "100px",
          height: "auto",
          objectFit: "contain",
          display: "block",
        }}
        src={`data:image/jpg;base64,${value["data"]}`}
        alt={key}
      />
    }
  </Box>
}

function FileDetail({ item }) {
  return (
    <Box sx={{
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
    }}>
      {item &&
        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          width: '100%',                  
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
    <Box>
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
        gridTemplateColumns: '1fr 2fr',
        gap: '1rem',
      }}
    >
       <FileList items={items} onSelectItem={handleSelectItem} />       
       <FileDetail item={selectedItem} />
    </Box>
  );
}