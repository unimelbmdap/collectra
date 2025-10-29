import { useState, useEffect } from 'react';
import { Box, Tab, Tabs } from '@mui/material';
import FileViewer from './file_viewer';

function CustomTabPanel({children, value, index}) {  
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      sx={{ width: '100%' }}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function Content({ items }) {

  const [value, setValue] = useState(0);

  const handleChange = (event, newValue) => {
    setValue(newValue);
  };
  
  return (
    <Box sx={{ border: 1, borderColor: 'divider', width: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={value} onChange={handleChange} aria-label="tabs menu">
          <Tab label="File" id={0}/>
          <Tab label="Workflow" id={1}/>          
        </Tabs>
      </Box>      
      <CustomTabPanel value={value} index={0}>
        <FileViewer items={items} />
      </CustomTabPanel>
      <CustomTabPanel value={value} index={1}>
        Item Two
      </CustomTabPanel>      
    </Box>
  );
}