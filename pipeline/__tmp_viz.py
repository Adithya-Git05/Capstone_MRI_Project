import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import nibabel as nib, os

BASE = r'c:\Users\Chinmai Datta Ram\Desktop\Capstone_Project'
mri   = nib.load(os.path.join(BASE,'data/UCSF-PDGM-0004/t1.nii.gz')).get_fdata().astype('f4')
risk  = nib.load(os.path.join(BASE,'outputs/V7_VALIDATED/enhanced_risk_map.nii.gz')).get_fdata().astype('f4')
corr  = nib.load(os.path.join(BASE,'outputs/V8_CORRIDORS/candidate_corridors.nii.gz')).get_fdata().astype('f4')
tumor = nib.load(os.path.join(BASE,'data/UCSF-PDGM-0004/tumor_mask.nii.gz')).get_fdata() > 0
cx,cy,cz = 162,154,105
bm = mri>0
p2,p98 = np.percentile(mri[bm],2), np.percentile(mri[bm],98)
mri_n = np.clip((mri-p2)/(p98-p2),0,1)
CORRIDORS = {1:('#FF3060','Lateral Frontal',0.0801),2:('#00CFFF','Superior Frontal',0.0736),3:('#FFE033','Anterior Frontal',0.0737),4:('#CC44FF','Pterional/Temporal',0.1287)}

PE_GLOW  = [pe.Stroke(linewidth=9,foreground='white',alpha=0.25), pe.Normal()]
PE_LABEL = [pe.Stroke(linewidth=2.5,foreground='black'), pe.Normal()]

def panel(ax, ms, rs, ts, cs, title):
    ax.set_facecolor('#0e1117')
    ax.imshow(np.rot90(ms), cmap='gray', vmin=0, vmax=0.9, interpolation='bilinear')
    rr = np.rot90(rs)
    masked = np.ma.masked_where(rr<0.15, rr)
    im = ax.imshow(masked, cmap='jet', alpha=0.50, vmin=0, vmax=1, interpolation='bilinear')
    ax.contour(np.rot90(ts.astype(float)), levels=[0.5], colors=['#22FF88'], linewidths=2.5, linestyles='dashed')
    for cid,(color,name,cost) in CORRIDORS.items():
        cmr = np.rot90((cs==cid).astype(float))
        if not np.any(cmr>0): continue
        # Draw filled region lightly to make corridor visible
        ax.contourf(cmr, levels=[0.5,1.5], colors=[color], alpha=0.25)
        # Draw outline
        ct = ax.contour(cmr, levels=[0.5], colors=[color], linewidths=5)
        # Apply path effects to each line artist in the contour
        for artist in ct.get_paths():
            pass  # paths not needed, we apply to ax.lines below
        # Use ax.plot for glow effect instead
        from matplotlib.path import Path
        for path in ct.get_paths():
            verts = path.vertices
            if len(verts) < 2: continue
            line, = ax.plot(verts[:,0], verts[:,1], color=color, linewidth=5, solid_capstyle='round', zorder=6)
            line.set_path_effects(PE_GLOW)
        ys,xs = np.where(cmr>0.5)
        if len(ys):
            by,bx = int(np.mean(ys)),int(np.mean(xs))
            ax.scatter(bx,by,color='white',s=350,zorder=9,alpha=0.4)
            ax.scatter(bx,by,color=color,s=150,zorder=10,edgecolors='white',linewidths=1.5)
            ax.annotate(f'C{cid}',xy=(bx,by),xytext=(bx+9,by-9),color='white',fontsize=9,fontweight='bold',zorder=11,path_effects=PE_LABEL)
    ax.set_title(title,color='white',fontsize=12,fontweight='bold',pad=8)
    ax.axis('off')
    return im

fig = plt.figure(figsize=(23,8.5),facecolor='#0e1117')
fig.subplots_adjust(left=0.03,right=0.84,top=0.82,bottom=0.13,wspace=0.06)
axs = [fig.add_subplot(1,3,i+1) for i in range(3)]
im = panel(axs[0],mri_n[cx,:,:],risk[cx,:,:],tumor[cx,:,:],corr[cx,:,:],f'Sagittal  (X={cx})')
panel(axs[1],mri_n[:,cy,:],risk[:,cy,:],tumor[:,cy,:],corr[:,cy,:],f'Coronal  (Y={cy})')
panel(axs[2],mri_n[:,:,cz],risk[:,:,cz],tumor[:,:,cz],corr[:,:,cz],f'Axial  (Z={cz})  -- Tumor Target')

cbar_ax = fig.add_axes([0.862,0.13,0.013,0.65])
cb = fig.colorbar(im,cax=cbar_ax)
cb.set_label('Model-derived surgical risk\n[0=Low, 1=Very High]',color='white',fontsize=9.5,labelpad=12)
cb.ax.yaxis.set_tick_params(color='white',labelsize=8)
plt.setp(plt.getp(cb.ax.axes,'yticklabels'),color='white')

patches=[mpatches.Patch(color=c,label=f'C{k}: {n}  (cost={co:.4f}, Pareto-eff.)') for k,(c,n,co) in CORRIDORS.items()]
patches.append(mpatches.Patch(color='#22FF88',label='Tumor boundary (green dashed)'))
fig.legend(handles=patches,loc='lower center',ncol=3,framealpha=0.22,facecolor='#1a2535',edgecolor='#44AAFF',fontsize=9.5,labelcolor='white',bbox_to_anchor=(0.43,0.00))

fig.suptitle('Patient: UCSF-PDGM-0004  |  V8 Candidate Surgical Corridors  -- Tri-Planar View\nBackground: T1 MRI anatomy  |  Overlay: risk heatmap  |  Coloured regions/lines: corridors\nRESEARCH PROTOTYPE  -- composite cost != clinical surgical safety',color='white',fontsize=12.5,fontweight='bold',y=0.97)

o1 = os.path.join(BASE,'outputs/V8_CORRIDORS/corridor_visualization.png')
o2 = os.path.join(BASE,'outputs/V7_VALIDATED/visualizations/corridor_overlay.png')
plt.savefig(o1,dpi=200,bbox_inches='tight',facecolor='#0e1117')
plt.savefig(o2,dpi=200,bbox_inches='tight',facecolor='#0e1117')
print('Saved',o1)
print('Saved',o2)
