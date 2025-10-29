import { useState, useEffect } from 'react';
import { Button, Container, Typography } from '@mui/material';
 
export default function App() {
  
  const [items, setItems] = useState([]);

  const getItems = async (type) => {
    let items = [];
    switch(type){
      case "single":
        items = await window.pywebview.api.getFile();
        break;
      case "folder":
        items = await window.pywebview.api.getFolder();
        break;
    }
    if (items) setItems(items);    
  }  

  useEffect(() => {
    console.log("Items:", items);
  }, [items]);
 
  return (
    <Container maxWidth="lg">
        <Typography variant="h1" gutterBottom>
          Collectra Editor
        </Typography>     
        <Button
          onClick={() => getItems("single")}
        >
          Open File
        </Button>
        <Button
          onClick={() => getItems("folder")}
          >
          Open Folder
        </Button>
    </Container>      
  );
}