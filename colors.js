  var Body = {
    setColor:function(color){
      document.querySelector('body').style.color = color;
    },
    setBackgroundColor:function(color){
      document.querySelector('body').style.backgroundColor = color;
    },
    setMode:function(name){
      document.querySelector('body').dataset.mode=name;
    }
  }
  var Links = {
    setColor:function(color){
      var links = document.getElementsByTagName('a');
      for(var i=0; i<links.length; i++) {
        links[i].style.color = color;
      }
    }
  }
  function nightDayHandler(self){
    var listNum = document.getElementsByTagName('ol')[0];
    if(self.value === 'day') {
      Body.setBackgroundColor('white');
      Body.setColor('black');
      Body.setMode('day');
      Links.setColor('black');
      listNum.style.color = 'black';
      self.value='night'
    } else {
      Body.setBackgroundColor('black');
      Body.setColor('lightgray');
      Body.setMode('night');
      Links.setColor('lightgray');
      listNum.style.color = 'lightgray';
      self.value='day';
    }
  }
