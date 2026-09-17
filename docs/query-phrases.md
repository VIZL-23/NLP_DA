# Query phrases

Every phrase below is one the named model was **trained on**, so each should
return a box on an image that actually shows that defect. 251 phrases across
23 class entries in the three shipped models.

## How to use this

- **Pick the model first.** A phrase only works with the model listed above it.
  Steel phrases do nothing on the concrete model, and the other way round.
- **Paste one phrase per line** into the description box. Several lines at once
  is fine — each gets its own colour.
- **Phrase 1 in each list is the one the model is scored on.** Start with it.
- **Keep the defect noun.** The model is keyed on the noun ("crack",
  "scratches", "spalling"). You can reword everything around it —
  `scratches on the metal surface` still works — but swap the noun for a
  synonym that is not in this list (`split`, `rupture`, `cleft`) and it usually
  returns nothing. See `phase-notes/PHASE-9.md` sections 7–8.
- **Nothing returned is not always a failure.** If the image does not show that
  defect, an empty result is the correct answer. Try the confidence slider
  before assuming the model is broken.

`crack` appears twice because it belongs to two models. The concrete-model list
omits three synonyms that were added after that model was trained.

Generated from `prompts/defect_corpus.json` by `python scripts/make_phrase_list.py`. If the corpus changes, regenerate
this file rather than editing it by hand.

---
## Model: Steel + concrete

Select **Steel + concrete** in the app. Default confidence **0.25**.

### Crazing

`neu_crack` · class `crazing` · AP@0.5 0.35 — weak class, expect misses

1. crazing on a steel surface
2. a network of fine cracks covering the metal surface
3. crazing
4. an irregular web of hairline cracks spreading across the surface
5. dense interconnected fine cracking with low contrast against the background
6. a mesh-like pattern of shallow surface cracks
7. crazing on hot-rolled steel strip
8. thermal crack network formed on the rolled steel surface
9. craze cracking
10. crack network defect
11. map cracking on the steel surface
12. a web of hairline fissures
13. crazed surface checking

### Inclusion

`neu_crack` · class `inclusion` · AP@0.5 0.72

1. an inclusion in the steel surface
2. foreign material embedded in the metal
3. inclusion
4. a dark elongated streak embedded in the steel surface
5. a narrow vertical dark band of impurity in the metal
6. an elongated dark blemish running along the rolling direction
7. non-metallic inclusion in hot-rolled steel
8. slag inclusion pressed into the steel strip during rolling
9. embedded impurity
10. non-metallic particle defect
11. embedded foreign matter in the steel
12. a sliver rolled into the surface
13. entrapped debris in the steel

### Patches

`neu_crack` · class `patches` · AP@0.5 0.89

1. surface patches on the steel
2. irregular discoloured areas on the metal surface
3. patches
4. irregular dark blotches spread over the steel surface
5. large uneven patches of darker tone against a lighter background
6. cloudy regions of contrasting shade on the metal
7. surface patch defect on hot-rolled steel strip
8. localised oxidation patches on the rolled surface
9. surface blotches
10. discolouration patch
11. blotches on the steel surface
12. irregular mottling across the steel
13. discoloured patches of surface

### Pitted surface

`neu_crack` · class `pitted_surface` · AP@0.5 0.86

1. a pitted steel surface
2. a surface covered in small pits and holes
3. pitted surface
4. densely scattered small cavities across the metal surface
5. a rough surface speckled with numerous tiny dark pits
6. widespread fine pitting giving a porous appearance
7. corrosion pitting on a steel plate
8. pitting damage across hot-rolled steel strip
9. corrosion pitting
10. pitting corrosion
11. a pockmarked steel surface
12. a cratered area of the steel
13. corrosion pits across the surface

### Rolled-in scale

`neu_crack` · class `rolled-in_scale` · AP@0.5 0.58

1. rolled-in scale on steel
2. oxide scale pressed into the metal surface
3. rolled-in scale
4. dark irregular flakes embedded into the steel surface
5. scattered dark scale fragments rolled flat into the surface
6. patchy dark oxide residue fused to the metal
7. rolled-in scale on hot-rolled steel strip
8. mill scale pressed into the strip during hot rolling
9. mill scale defect
10. embedded oxide scale
11. scale pressed into the steel surface
12. embedded mill scale
13. rolled-in oxide flakes

### Scratches

`neu_crack` · class `scratches` · AP@0.5 0.90

1. scratches on the steel surface
2. long thin scratch marks on the metal
3. scratches
4. a long straight bright line running across the surface
5. narrow elongated grooves scored into the metal
6. thin high-contrast linear streaks along the rolling direction
7. mechanical scratch on hot-rolled steel strip
8. abrasion scoring from roller contact
9. surface scoring
10. abrasion mark
11. gouges in the metal surface
12. furrows worn into the steel surface
13. drag marks across the metal

### Crack

`neu_crack` · class `crack` · AP@0.5 0.44

1. a crack in the concrete surface
2. a thin fracture line running across the pavement
3. crack
4. a long dark jagged line splitting the surface
5. a narrow branching fissure winding across the material
6. a thin high-contrast fracture against a textured background
7. structural crack in a concrete surface
8. fatigue crack in asphalt pavement
9. surface fissure
10. fracture line
11. a split in the concrete surface
12. a fissure running through the concrete
13. a fracture line in the concrete

---

## Model: Steel · GC10-DET

Select **Steel · GC10-DET** in the app. Default confidence **0.25**.

### Crease

`gc10` · class `crease` · AP@0.5 0.46

1. a crease in the steel sheet
2. a folded line running across the metal strip
3. crease
4. a sharp straight fold line across the sheet surface
5. a narrow horizontal band where the metal has been folded
6. a linear ridge left by the sheet doubling over on itself
7. crease defect on cold-rolled steel sheet
8. fold mark formed while coiling the steel strip
9. fold line
10. sheet crease mark

### Crescent gap

`gc10` · class `crescent_gap` · AP@0.5 0.94

1. a crescent-shaped gap at the strip edge
2. a curved notch missing from the edge of the steel sheet
3. crescent gap
4. a smooth half-moon cutout along the sheet border
5. a dark curved bite taken out of the metal edge
6. an arc-shaped void interrupting the straight strip edge
7. crescent gap on cold-rolled steel strip
8. edge break formed during shearing of the steel sheet
9. moon-shaped edge defect
10. edge notch

### Inclusion (GC10)

`gc10` · class `gc10_inclusion` · AP@0.5 0.16 — weak class, expect misses

1. an inclusion on the steel sheet
2. a foreign particle pressed into the sheet surface
3. steel sheet inclusion
4. a small dark speck embedded in the rolled sheet
5. isolated dark inclusions scattered on a bright sheet surface
6. compact dark blemishes set into the metal
7. non-metallic inclusion on cold-rolled steel sheet
8. impurity rolled into the sheet during production
9. foreign body defect
10. sheet impurity

### Oil spot

`gc10` · class `oil_spot` · AP@0.5 0.57

1. an oil spot on the steel sheet
2. an oily stain on the metal surface
3. oil spot
4. a dark smudged stain with soft irregular edges
5. a glossy dark residue smeared across the sheet
6. a diffuse dark blotch with no sharp boundary
7. lubricant oil contamination on cold-rolled steel
8. rolling oil residue left on the strip surface
9. oil stain
10. grease mark

### Punching hole

`gc10` · class `punching_hole` · AP@0.5 0.98

1. a punched hole in the steel sheet
2. a small round hole through the metal
3. punching hole
4. a sharply defined dark circular opening in the sheet
5. a small black round void with a clean edge
6. an isolated perforation piercing the metal surface
7. punch hole defect on cold-rolled steel sheet
8. hole left by the punching press during processing
9. perforation
10. punched opening

### Rolled pit

`gc10` · class `rolled_pit` · AP@0.5 0.20 — weak class, expect misses

1. a rolled pit on the steel sheet
2. a small indentation pressed into the metal
3. rolled pit
4. a shallow dark depression stamped into the surface
5. a compact pitted dent repeating along the strip
6. a small crater-like mark impressed in the sheet
7. roll-induced pit on cold-rolled steel sheet
8. indentation transferred from a damaged work roll
9. roll mark pit
10. pressed indentation

### Silk spot

`gc10` · class `silk_spot` · AP@0.5 0.62

1. a silk spot on the steel sheet
2. faint silky streaks on the metal surface
3. silk spot
4. pale wispy streaks with a fibrous silk-like texture
5. soft elongated light marks running along the rolling direction
6. delicate low-contrast striations across the sheet
7. silk spot defect on cold-rolled steel sheet
8. fibrous surface marking from uneven rolling
9. silky streak
10. fibre-like surface mark

### Waist folding

`gc10` · class `waist_folding` · AP@0.5 0.85

1. waist folding on the steel sheet
2. wrinkled folds across the middle of the strip
3. waist folding
4. a series of parallel fold lines banding across the sheet
5. repeated wrinkle ridges running transverse to the strip
6. a corrugated band of folds through the sheet centre
7. waist fold defect on cold-rolled steel strip
8. transverse buckling formed during strip tensioning
9. waist wrinkle
10. transverse fold band

### Water spot

`gc10` · class `water_spot` · AP@0.5 0.67

1. a water spot on the steel sheet
2. a dried water stain on the metal surface
3. water spot
4. a pale irregular stain with a faint outline
5. a light streaky residue left by evaporated liquid
6. a soft-edged blotch slightly lighter than the surrounding metal
7. water stain on cold-rolled steel sheet
8. residue from incomplete drying after strip cleaning
9. water mark
10. moisture stain

### Welding line

`gc10` · class `welding_line` · AP@0.5 0.92

1. a welding line across the steel strip
2. a weld seam joining two lengths of sheet
3. welding line
4. a long straight horizontal seam spanning the full sheet width
5. a continuous narrow band crossing the entire strip
6. a raised linear join running edge to edge
7. coil joining weld on cold-rolled steel strip
8. butt weld seam between two steel coils
9. weld seam
10. strip joint weld

---

## Model: Concrete · structural

Select **Concrete · structural** in the app. Default confidence **0.10**.

AP is low across this model because its boxes are loose field photography, not because it cannot find defects: at the 0.10 default it hit 80% of test images, including efflorescence 6/6 and spalling 6/6. Lead with **spalling**.

### Exposed reinforcement

`concrete` · class `exposed_reinforcement` · AP@0.5 0.32 — weak class, expect misses

1. exposed reinforcement bars in the concrete
2. steel rebar visible through the concrete
3. exposed reinforcement
4. parallel metal bars showing through a broken surface
5. rusted rods protruding from the material
6. exposed metal ribbing inside a cavity
7. exposed reinforcement in a concrete structure
8. corroded steel reinforcement in a concrete member
9. exposed rebar
10. visible reinforcement

### Rust stain

`concrete` · class `rust_stain` · AP@0.5 0.03 — **does not work, avoid in a demo**

1. rust staining on the concrete surface
2. brown rust marks running down the concrete
3. rust stain
4. reddish-brown streaks discolouring the surface
5. an orange stain bleeding across the material
6. rust-coloured runoff marks on a grey surface
7. corrosion staining on a concrete structure
8. iron oxide staining on cementitious material
9. corrosion stain
10. rust discolouration

### Scaling

`concrete` · class `scaling` · AP@0.5 0.31 — weak class, expect misses

1. scaling on the concrete surface
2. the surface layer of concrete peeling away
3. scaling
4. a shallow flaked area across the surface
5. a patchy loss of the outer layer
6. a roughened region where the finish has worn off
7. scaling on a concrete slab
8. surface mortar loss on cementitious material
9. surface peeling
10. concrete scaling

### Spalling

`concrete` · class `spalling` · AP@0.5 0.58

1. spalling on the concrete surface
2. concrete broken away from the surface
3. spalling
4. a crater where material has flaked off the surface
5. a rough patch where the surface layer has broken away
6. an irregular depression exposing coarse aggregate
7. spalled concrete on a structural element
8. surface fragmentation of a concrete member
9. concrete break-off
10. surface flaking

### Crack

`concrete` · class `crack` · AP@0.5 0.38 — weak class, expect misses

1. a crack in the concrete surface
2. a thin fracture line running across the pavement
3. crack
4. a long dark jagged line splitting the surface
5. a narrow branching fissure winding across the material
6. a thin high-contrast fracture against a textured background
7. structural crack in a concrete surface
8. fatigue crack in asphalt pavement
9. surface fissure
10. fracture line

### Efflorescence

`concrete` · class `efflorescence` · AP@0.5 0.33 — weak class, expect misses

1. efflorescence on the concrete surface
2. white salt deposits leaching out of the concrete
3. efflorescence
4. a chalky white bloom spreading across the surface
5. pale crystalline staining seeping from the material
6. a white powdery patch on a grey surface
7. efflorescence on a concrete structure
8. salt crystallisation on cementitious material
9. salt bloom
10. lime staining

---

