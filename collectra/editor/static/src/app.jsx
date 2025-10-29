import { useState, useEffect } from 'react';
import { Button, Container, Box, Typography } from '@mui/material';
import Content from './components/content.jsx';


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
    <Container minWidth="lg" sx={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: "1.5rem"      
    }}>      
      <Typography variant="h1" gutterBottom fontSize={24}>
        Collectra Editor
      </Typography>             
      <Box sx={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        gap: "1rem"
      }}>
        <Button
          onClick={() => getItems("single")}
          variant="outlined"
        >
          Open File
        </Button>
        <Button
          onClick={() => getItems("folder")}
          variant="outlined"
        >
          Open Folder
        </Button>                
      </Box>
      <Content items={items} />
    </Container>      
  );
}