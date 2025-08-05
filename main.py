import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def dotProduct(a,b):
  return np.einsum('...ij,j->...i', a, b)


st.title("Optical Lattice potential viewer")
st.write("This web app allows you to vizualize the various properties of different optical lattices.")

gridSize = st.container()
x = gridSize.slider("Lattice Resolution", min_value = 100, max_value = 1000, value = 300, step = 1)
coords = np.meshgrid(np.arange(int(x))/10, np.arange(int(x))/10)

kagome, honeycomb, tab3 = st.tabs(["Kagome", "Honeycomb", "Lieb"])

t1col1, t1col2 = kagome.columns([1,2])

#Kagome
b1 = np.array([3/2,-np.sqrt(3)/2])
b2 = np.array([0,np.sqrt(3)])
q1 = b1
q2 = b2
q3 = (b1+b2)
def LWLattice(x,y,a,b,scale):
  coords = np.stack([x+a, y+b], axis=-1)/scale
  return (np.cos(dotProduct(coords,q1)) + np.cos(dotProduct( coords, q2)) + np.cos(dotProduct( coords, q3)))
vSW = t1col1.slider("ShortWave Lattice Strength",min_value = 0.0, max_value = 10.0, value = 1.0, step = 0.1)
vLW = t1col1.slider("Long Wave Lattice Strength",min_value = 0.0, max_value = 10.0, value = 0.8, step = 0.1)
a =t1col1.slider("LW Lattice x offset",min_value = -50.0, max_value = 50.0, value = 0.0, step = 0.1)
b= t1col1.slider("LW Lattice y offset",min_value = -50.0, max_value = 50.0, value = 0.0, step = 0.1)
scale = t1col1.slider("LW Lattice scale",min_value = 0.0, max_value = 10.0, value = 2.0, step = 0.1)
shortWaveLattice = vSW*LWLattice(coords[0], coords[1],0,0,1)
longWaveLattice = vLW*LWLattice(coords[0], coords[1],a,b,scale)
result = shortWaveLattice + longWaveLattice
plt.style.use('bmh')
plt.imshow(result,cmap='jet')
plt.colorbar(label='Potential Depth (a.u.)')
plt.title("Kagome Lattice")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
t1col2.pyplot(plt.gcf())

plt.close()
plt.plot(-result[0])
plt.xlim(0,250)
plt.ylim()
plt.title("Kagome Lattice Cross-Section")
t1col2.pyplot(plt.gcf())

#honeycomb
t2col1, t2col2 = honeycomb.columns([1,2])
VX1 = t2col1.slider("Auxillary X Field Strength", min_value = 0.0, max_value = 8.0, value = 8.0, step = 0.1)
VX = t2col1.slider("X Field Strength", min_value = 0.0, max_value = 1.0, value = 1.0, step = 0.1)
VY = t2col1.slider("Y Field Strength", min_value = 0.0, max_value = 10.0, value = 2.0, step = 0.1)
k = 2*np.pi/(t2col1.slider("wavelength", min_value = 350.0, max_value = 1240.0, value = 500.0, step = 0.1)/100)
theta = t2col1.slider("Theta", min_value = -np.pi, max_value = np.pi, value = np.pi/4, step = 0.1)
phi = t2col1.slider("phi", min_value = -np.pi, max_value = np.pi, value = np.pi/4, step = 0.1)
alpha = t2col1.slider("alpha", min_value = 0.0, max_value = 10.0, value = 1.0, step = 0.1)
def honeycombLattice(x,y):
  return (-VX1*np.cos(k*x + theta/2)**2 - VX*np.cos(k*x)**2 - VY*np.cos(k*y)**2 - 2*alpha*np.sqrt(VX*VY)*np.cos(k*x)*np.cos(k*y)*np.cos(phi))
result = honeycombLattice(coords[0], coords[1])
plt.imshow(-result,cmap='jet')
plt.colorbar(label='Potential Depth (a.u.)')
plt.title("Honeycomb Lattice")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
t2col2.pyplot(plt.gcf())