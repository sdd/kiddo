#!/usr/bin/env just --justfile

test-donnelly:
    cargo test donnelly

bench-d-v2:
    cargo bench --bench donnelly_v2

bench-d-v2b:
    cargo bench --bench donnelly_v2_branchless

# Generate x86-64-v4 assembly for donnelly functions in Compiler Explorer
godbolt-x86-v4:
    #!/usr/bin/env bash
    set -e
    
    # Read the entire file and add #[no_mangle] to all pub fn
    SOURCE=$(cat src/donnelly_stem_layout.rs | sed 's/^pub fn/#[no_mangle]\
pub fn/g')
    
    # URL encode the source
    ENCODED=$(echo "$SOURCE" | python3 -c "
import sys, urllib.parse
content = sys.stdin.read()
print(urllib.parse.quote(content, safe=''))")
    
    # Open in Compiler Explorer
    open "https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:rust,selection:(endColumn:2,endLineNumber:25,positionColumn:2,positionLineNumber:25,selectionStartColumn:2,selectionStartLineNumber:25,startColumn:2,startLineNumber:25),source:'$ENCODED'),l:'5',n:'0',o:'Rust+source+%231',t:'0')),k:33.333333333333336,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:r1700,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:rust,libs:!(),options:'-C+target-cpu%3Dx86-64-v4+-C+opt-level%3D3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+rustc+1.70.0+(Editor+%231)',t:'0')),k:66.66666666666667,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4"

# Generate Apple M2 assembly for donnelly functions in Compiler Explorer  
godbolt-m2:
    #!/usr/bin/env bash
    set -e
    
    # Read the entire file and add #[no_mangle] to all pub fn
    SOURCE=$(cat src/donnelly_stem_layout.rs | sed 's/^pub fn/#[no_mangle]\
pub fn/g')
    
    # URL encode the source
    ENCODED=$(echo "$SOURCE" | python3 -c "
import sys, urllib.parse
content = sys.stdin.read()
print(urllib.parse.quote(content, safe=''))")
    
    # Open in Compiler Explorer
    open "https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:rust,selection:(endColumn:2,endLineNumber:25,positionColumn:2,positionLineNumber:25,selectionStartColumn:2,selectionStartLineNumber:25,startColumn:2,startLineNumber:25),source:'$ENCODED'),l:'5',n:'0',o:'Rust+source+%231',t:'0')),k:33.333333333333336,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:aarch64-rustc-1700,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:rust,libs:!(),options:'-C+target-cpu%3Dapple-m2+-C+opt-level%3D3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+aarch64+rustc+1.70.0+(Editor+%231)',t:'0')),k:66.66666666666667,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4"

# Generate x86-64-v4 assembly for donnelly_get_idx_v2
asm-x86-v4:
    RUSTFLAGS="-C target-cpu=znver3 -C opt-level=3" \
    cargo rustc --lib --release -- --emit asm -o target/donnelly_get_idx_v2_x86_64_v4.s
    @echo "Assembly output written to target/donnelly_get_idx_v2_x86_64_v4.s"
    @echo "Search for 'donnelly_get_idx_v2' in the file to find the function"

# Generate Apple M2 assembly for donnelly_get_idx_v2
asm-m4:
    RUSTFLAGS="-C target-cpu=apple-m4 -C opt-level=3" \
    cargo rustc --lib --release --features no_inline -- --emit asm -o target/donnelly_get_idx_v2_apple_m2.s
    @echo "Assembly output written to target/donnelly_get_idx_v2_apple_m2.s"
    @echo "Search for 'donnelly_get_idx_v2' in the file to find the function"