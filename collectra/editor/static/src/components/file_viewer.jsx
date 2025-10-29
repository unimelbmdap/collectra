import { useState } from "react";
import { Box, List, ListItem } from "@mui/material";

function FileDetail({ item }) {
  return (
    <Box>
      {
        item && <h1>{item.key}</h1>
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